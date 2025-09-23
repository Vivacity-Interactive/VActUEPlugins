#include "CameraPlayerController.h"
#include "CineCameraComponent.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerInput.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/PawnMovementComponent.h"
#include "GameFramework/CharacterMovementComponent.h"

ACameraPlayerController::ACameraPlayerController()
	: bInvertY(false)
	, bUseCameraHints(true)
	, bUseCameraSampleRadius(false)
	, bSetCurrentTargetBeginPlay(true)
	, bCameraEnableCollision(false)
	, bUseCameraTrackOffset(false)
	, bRPCOrientRotationToMovement(true)
	, RPCTagName(NAME_None)
	, FocusSocketName(NAME_None)
	, FollowSocketName(NAME_None)
	, TrackSocketName(NAME_None)
	, RPC(nullptr)
	, CameraPawn(nullptr)
	, CameraSample(nullptr)
	, InputActions()
	, CameraHintRadius(300.0f)
	, CameraHintDamp(0.8f)
	, CameraHintDampTarget(0.8f)
	, CameraHintDampSpeed(0.4f)
	, MaxOperatorDampVelocity(0.1f)
	, ZoomSensitivityScale(1.0f)
	, LookSensitivityScale(1.0f)
	, HintLocationAlpha(FVector::OneVector)
	, HintRotationAlpha(FVector::OneVector)
	, CameraCurrent()
	, CameraTarget()
	, OperatorCurrent()
	, OperatorVelocity(0.0f)
	, OperatorTarget()
	, OperatorDamp(1.0f)
{
	bShowMouseCursor = false;
	DefaultMouseCursor = EMouseCursor::Default;

	OperatorDamp.FocalLengt = 1.0f;
	OperatorDamp.Follow = FVector(0.01f, 0.05f, 1.0f);
	OperatorDamp.Track = FRotator(1.0f);
}

void ACameraPlayerController::BeginPlay()
{
	Super::BeginPlay();

	UEnhancedInputLocalPlayerSubsystem* Subsystem;

	bool bSubSystem = (Subsystem = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(GetLocalPlayer())) != nullptr;
	if (bSubSystem) { Subsystem->AddMappingContext(InputActions.DefaultMappingContext, 0); }

	if (bSetCurrentTargetBeginPlay)
	{
		CameraCurrent = CameraTarget;
		OperatorCurrent = OperatorTarget;
		CameraPawn->AssignCameraOperatorVector(OperatorCurrent);
		CameraPawn->AssignCameraVector(CameraCurrent);
		OperatorDamp.Follow = CameraCurrent.FollowSensitivity;
		FVActCamera::AssignInto(OperatorDamp.Track, CameraCurrent.TrackSensitivity);
	}
}

void ACameraPlayerController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	const bool bCamera = CameraPawn != nullptr && CameraPawn.IsValid();
	const bool bRPC = RPC != nullptr && RPC.IsValid();
	const bool bCameraHints = bUseCameraHints && bRPC;

	bool bNewHint = false;

	if (bCamera)
	{
		UCineCameraComponent* CineCameraComponent = CameraPawn->GetCineCameraComponent();

		CameraPawn->SetActorEnableCollision(bCameraEnableCollision);
		CameraPawn->ExtractCameraVector(CameraCurrent);
		CameraPawn->ExtractCameraOperatorVector(OperatorCurrent);

		if (bRPC)
		{
			const FVector FollowTo = RPC->GetRootComponent()->GetSocketLocation(FollowSocketName) + CameraCurrent.FollowOffset * CameraCurrent.FollowOffsetFactor;
			const FVector FollowFrom = CineCameraComponent->GetComponentLocation();
			CameraTarget.Location = OperatorTarget.Follow = FollowTo;

			const FVector From = CineCameraComponent->GetComponentLocation();
			const FVector TrackTo = RPC->GetRootComponent()->GetSocketLocation(TrackSocketName);
			CameraTarget.Rotation = OperatorTarget.Track = (TrackTo - From).Rotation() + FVActCamera::RWeight(CameraCurrent.TrackOffset, CameraCurrent.TrackOffsetFactor) * bUseCameraTrackOffset;
		}

		if (bCameraHints)
		{
			const bool bSample = CameraSample != nullptr && CameraSample.IsValid();
			FVector RPCLocation = RPC->GetActorLocation();

			UWorld* World = GetWorld();
			int32 Index = 0;
			float Score = bSample ? FMath::Abs(FVector::Dist(RPCLocation, CameraSample->GetActorLocation())) : FLT_MAX;

			for (TActorIterator<ACameraSample> It(World); It; ++It, ++Index)
			{
				ACameraSample* Sample = *It;
				const bool bCheck = Sample->Hint != nullptr && Sample->Hint.IsValid();
				const float ScoreTest = FMath::Abs(FVector::Distance(RPCLocation, Sample->GetActorLocation()));

				const bool bPick = ScoreTest < Score && (!bUseCameraSampleRadius || ScoreTest <= CameraHintRadius);
				if (bPick)
				{
					bNewHint = !bSample || Sample->Hint != CameraSample->Hint;
					if (bNewHint) { CameraHintDamp = CameraHintProgress = 0; }
					CameraSample = Sample;
				}
			}

			const bool bMatchHint = CameraSample != nullptr && CameraSample.IsValid() && !FMath::IsNearlyEqual(CameraHintProgress, 1.0f);
			if (bMatchHint)
			{
				CameraTarget = CameraSample->Hint->Vector;

				const bool bFollowOffset = bRPC && bNewHint;
				if (bFollowOffset)
				{
					const FVector From = CameraSample->Hint->GetActorLocation();
					const FVector FollowTo = RPC->GetRootComponent()->GetSocketLocation(FollowSocketName);
					CameraTarget.FollowOffset = From - FollowTo;
					CameraHintDampTarget = CameraSample->Damp;
				}

				CameraTarget.Location = OperatorTarget.Follow = FMath::Lerp(OperatorTarget.Follow, CameraTarget.Location, HintLocationAlpha); //CameraTarget.Alpha
				CameraTarget.Rotation = OperatorTarget.Track = FVActCamera::RLerp(OperatorTarget.Track, CameraTarget.Rotation, HintRotationAlpha); //CameraTarget.Alpha

				CameraHintDamp = FMath::FInterpTo(CameraHintDamp, CameraHintDampTarget, DeltaTime, CameraHintDampSpeed);
				CameraHintProgress = FMath::FInterpTo(CameraHintProgress, 1.0f, DeltaTime, CameraHintDamp);

				FVActCamera::Interp(CameraCurrent, CameraCurrent, CameraTarget, DeltaTime, CameraHintDamp);
				OperatorDamp.Follow = CameraCurrent.FollowSensitivity;
				FVActCamera::AssignInto(OperatorDamp.Track, CameraCurrent.TrackSensitivity);
			}
		}

		const bool bSample = CameraSample != nullptr && CameraSample.IsValid() && CameraSample->Hint != nullptr && CameraSample->Hint.IsValid();
		if (bSample)
		{
			FVector HintLocation = CameraSample->Hint->Vector.Location;
			FVector LimitedFollowDistance = FVActCamera::VClamp(OperatorTarget.Follow, HintLocation - CameraCurrent.FollowDistanceMax, HintLocation + CameraCurrent.FollowDistanceMax);
			OperatorTarget.Follow = FMath::Lerp(OperatorTarget.Follow, LimitedFollowDistance, CameraCurrent.FollowDistanceMaxAlpha);

			FRotator HintRotation = CameraSample->Hint->Vector.Rotation;
			FRotator LimitedTrackAngle = FVActCamera::RClamp(OperatorTarget.Track, HintRotation - CameraCurrent.TrackAngleMax, HintRotation + CameraCurrent.TrackAngleMax);
			OperatorTarget.Track = FVActCamera::RLerp(OperatorTarget.Track, LimitedTrackAngle, CameraCurrent.TrackAngleMaxAlpha);
		}		

		FVActCamera::Interp(OperatorCurrent, OperatorCurrent, OperatorTarget, OperatorVelocity, OperatorDamp, MaxOperatorDampVelocity, DeltaTime);

		CameraPawn->AssignCameraOperatorVector(OperatorCurrent);
		CameraPawn->AssignCameraVector(CameraCurrent);
		
	}

}

void ACameraPlayerController::OnPossess(APawn* InPawn)
{
	Super::OnPossess(InPawn);
	
	const bool bValid = InPawn->IsA<ACameraPawn>();
	if (!bValid)
	{
		UnPossess();
		const bool bCamera = CameraPawn != nullptr && CameraPawn.IsValid();
		if (bCamera) { Possess(CameraPawn.Get()); }
	}

	CameraPawn = Cast<ACameraPawn>(GetPawn());

	const bool bCamera = CameraPawn != nullptr && CameraPawn.IsValid();
	if (bCamera)
	{
		ResolveRPC();
		InitForNewNoirCamera();
	}
}

void ACameraPlayerController::ResolveRPC(APawn* InRPC)
{
	AActor* BestRPC = nullptr;
	UWorld* World = GetWorld();

	const bool bInRPC = InRPC != nullptr;
	if (bInRPC) { RPC = InRPC; }

	bool bRPC = RPC != nullptr && RPC.IsValid();
	if (!bRPC)
	{
		const bool bByTag = RPCTagName != NAME_None;
		if (bByTag)
		{
			for (TActorIterator<APawn> It(World); It; ++It)
			{
				APawn* PickActor = *It;
				if (PickActor->ActorHasTag(RPCTagName)) { RPC = PickActor; break; }
			}
		}

		const bool bByClass = RPC == nullptr || !RPC.IsValid();
		if (bByClass)
		{
			AActor* PickActor = UGameplayStatics::GetActorOfClass(World, APawn::StaticClass());
			RPC = Cast<APawn>(PickActor);
		}
	}

	bRPC = RPC != nullptr && RPC.IsValid();
	if (bRPC)
	{ 
		InitForNewRPC();
	}
}

void ACameraPlayerController::InitForNewRPC()
{
	const bool bCamera = CameraPawn != nullptr && CameraPawn.IsValid();
	if (bCamera)
	{
		UCineCameraComponent* CineCameraComponent = CameraPawn->GetCineCameraComponent();
		FTransform const FocusSocketTransform = RPC->GetRootComponent()->GetSocketTransform(FocusSocketName, RTS_Actor);
		CineCameraComponent->FocusSettings.TrackingFocusSettings.ActorToTrack = RPC;
		CineCameraComponent->FocusSettings.TrackingFocusSettings.RelativeOffset = FocusSocketTransform.GetLocation();
		CineCameraComponent->FocusSettings.bSmoothFocusChanges = true;
		CineCameraComponent->FocusSettings.FocusMethod = ECameraFocusMethod::Tracking;
	}
}

void ACameraPlayerController::InitForNewNoirCamera()
{
	const bool bRPC = RPC != nullptr && RPC.IsValid();
	if (bRPC)
	{
		UCineCameraComponent* CineCameraComponent = CameraPawn->GetCineCameraComponent();

		const FVector From = CineCameraComponent->GetComponentLocation();
		const FVector FollowTo = RPC->GetRootComponent()->GetSocketLocation(FollowSocketName);
		CameraCurrent.FollowOffset = CameraTarget.FollowOffset = From - FollowTo;
		OperatorCurrent.Follow = OperatorTarget.Follow = From;

		const FVector TrackTo = RPC->GetRootComponent()->GetSocketLocation(TrackSocketName);
		OperatorCurrent.Track = OperatorTarget.Track = (TrackTo - From).Rotation();

		CameraPawn->SetActorEnableCollision(bCameraEnableCollision);
		CameraPawn->AssignCameraVector(CameraCurrent);
	}
}

void ACameraPlayerController::SetupInputComponent()
{
	Super::SetupInputComponent();

	if (UEnhancedInputComponent* EnhancedInputComponent = Cast<UEnhancedInputComponent>(InputComponent))
	{
		// Moving
		EnhancedInputComponent->BindAction(InputActions.MoveAction, ETriggerEvent::Triggered, this, &ACameraPlayerController::Move);

		// Looking
		EnhancedInputComponent->BindAction(InputActions.LookAction, ETriggerEvent::Triggered, this, &ACameraPlayerController::Look);

		// Zooming
		EnhancedInputComponent->BindAction(InputActions.ZoomAction, ETriggerEvent::Triggered, this, &ACameraPlayerController::Zoom);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("'%s' Failed to find an Enhanced Input component!"), *GetNameSafe(this));
	}
}

// Callback input functions

void ACameraPlayerController::Move_Implementation(const FInputActionValue& Value)
{
	UPawnMovementComponent* MovementComponent = nullptr;
	UCharacterMovementComponent* CharacterMovementComponent;

	const bool bRPC = RPC != nullptr && RPC.IsValid() && (MovementComponent = RPC->GetMovementComponent()) != nullptr;
	if (bRPC)
	{
		FVector Movement = FVector(Value.Get<FVector2D>(), 0);
		FRotator Rotation = GetControlRotation();

		const FRotator YawRotation(0, Rotation.Yaw, 0);

		const FVector ForwardDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
		const FVector RightDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
		
		const bool bCharacterMovement = (CharacterMovementComponent = Cast<UCharacterMovementComponent>(MovementComponent)) != nullptr;
		if (bCharacterMovement) { CharacterMovementComponent->bOrientRotationToMovement = bRPCOrientRotationToMovement; }

		MovementComponent->AddInputVector(ForwardDirection * Movement.Y);
		MovementComponent->AddInputVector(RightDirection * Movement.X);
	}
}

void ACameraPlayerController::Look_Implementation(const FInputActionValue& Value)
{
	FVector2D LookAxisVector = Value.Get<FVector2D>();

	const bool bRPC = RPC != nullptr && RPC.IsValid();
	if (bRPC)
	{
		const FRotator DeltaLook = FRotator(LookAxisVector.X, LookAxisVector.Y, 0);
		CameraTarget.TrackOffset += DeltaLook * LookSensitivityScale;
	}
}

void ACameraPlayerController::Zoom_Implementation(const FInputActionValue& Value)
{
	float ZoomValue = Value.Get<float>();

	OperatorTarget.FocalLengt += ZoomValue * CameraCurrent.FocalLengthSensitivity * ZoomSensitivityScale;
	OperatorTarget.FocalLengt = FMath::Clamp(OperatorTarget.FocalLengt, CameraCurrent.FocalLengthMin, CameraCurrent.FocalLengthMax);
}

