#include "VRPlayerController.h"

#include "Engine/EngineTypes.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputActionValue.h"
#include "XRLoadingScreenFunctionLibrary.h"
#include "HAL/IConsoleManager.h"
#include "Kismet/GameplayStatics.h"
#include "NavigationSystem.h"

AVRPlayerController::AVRPlayerController()
{
	SecondaryScreenPercentage = 100.0f;
	TrackOrigin = EHMDTrackingOrigin::Type::Stage;
	TraceLaunchVelocity = 650.0f;
	bTracePath = true;
	bTraceComplex = false;
	bTraceUseNavMesh = true;
	bMoveValid = false;
	bTracePathShow = false;
	SnapTurnDegree = -45.0f;
	NavMeshCellHeight = 8.0f;

#if WITH_EDITORONLY_DATA
	_DEBUG_Show_Draw_Limits = false;
	_DEBUG_Line_Width = 0.8f;
#endif
}

//void AVRController::InitForNewRPC()
//{
//
//}

void AVRPlayerController::InitForNewVRCamera()
{

}

bool AVRPlayerController::IsValidMoveLocation(const FHitResult& Hit, FVector& ProjectedLocation)
{
	bool bResult = false;
	bVRDynamic = bResult = Hit.Component.IsValid() && Hit.Component->GetCollisionObjectType() == DynamicObjectChannel;
	if (!bVRDynamic)
	{
		FVector QueryExtent = FVector::Zero();
		FNavLocation OutNavLocation;
		ANavigationData* UseNavData = nullptr;
		UNavigationSystemV1* NavSystem = UNavigationSystemV1::GetNavigationSystem(GetWorld());
		const bool bNavigate = NavSystem != nullptr && (UseNavData = NavSystem->GetDefaultNavDataInstance(FNavigationSystem::DontCreate)) != nullptr;
		if (bNavigate)
		{
			bResult = NavSystem->ProjectPointToNavigation(Hit.Location, OutNavLocation, QueryExtent, UseNavData);
			ProjectedLocation = OutNavLocation.Location;
		}
	}
	else
	{
		ProjectedLocation = Hit.Location;
		VRDynamicComponent = Hit.Component;
	}

	return bResult;
}


void AVRPlayerController::SnapTurn(bool bRight)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		UCineCameraComponent* CineCamera = VRCamera->GetCineCamera();
		
		float YawDelta = bRight ? FMath::Abs(SnapTurnDegree) : SnapTurnDegree;
		FQuat DeltaRotation(FRotator(0.0, YawDelta, 0.0));
		FVector Location = VRCamera->GetActorLocation();

		FQuat NewRotation = VRCamera->GetActorQuat() * DeltaRotation;
		FTransform NewTransform = CineCamera->GetRelativeTransform() * FTransform(NewRotation, Location);
		FVector NewLocation = (CineCamera->GetComponentLocation() - NewTransform.GetLocation()) + Location;
		
		VRCamera->AddActorWorldRotation(DeltaRotation);
		VRCamera->SetActorLocation(NewLocation);
	}
}

void AVRPlayerController::SetupInputComponent()
{
	Super::SetupInputComponent();

	if (UEnhancedInputComponent* EnhancedInputComponent = Cast<UEnhancedInputComponent>(InputComponent))
	{
		// Moving
		EnhancedInputComponent->BindAction(MoveAction, ETriggerEvent::Triggered, this, &AVRPlayerController::Move);
		EnhancedInputComponent->BindAction(MoveAction, ETriggerEvent::Started, this, &AVRPlayerController::MoveStart);
		EnhancedInputComponent->BindAction(MoveAction, ETriggerEvent::Completed, this, &AVRPlayerController::MoveStop);

		// Looking
		//EnhancedInputComponent->BindAction(LookAction, ETriggerEvent::Triggered, this, &AVRController::Look);

		// Orientating
		EnhancedInputComponent->BindAction(TurnAction, ETriggerEvent::Triggered, this, &AVRPlayerController::Turn);

		// Grabing Left
		EnhancedInputComponent->BindAction(GrabLeftAction, ETriggerEvent::Triggered, this, &AVRPlayerController::GrabLeft);
		EnhancedInputComponent->BindAction(GrabLeftAction, ETriggerEvent::Completed, this, &AVRPlayerController::GrabLeftStop);

		// Grabing Right
		EnhancedInputComponent->BindAction(GrabRightAction, ETriggerEvent::Triggered, this, &AVRPlayerController::GrabRight);
		EnhancedInputComponent->BindAction(GrabRightAction, ETriggerEvent::Completed, this, &AVRPlayerController::GrabRightStop);
	}
	else
	{
		//LogTemplateCharacter
		UE_LOG(LogTemp, Error, TEXT("'%s' Failed to find an Enhanced Input component! This template is built to use the Enhanced Input system. If you intend to use the legacy system, then you will need to update this C++ file."), *GetNameSafe(this));
	}
}

void AVRPlayerController::BeginPlay()
{
	Super::BeginPlay();

	//TraceObjectTypes.Add(UEngineTypes::ConvertToObjectType(DynamicObjectChannel));

	bool bValid = UHeadMountedDisplayFunctionLibrary::IsHeadMountedDisplayEnabled();
	if (bValid)
	{
		UEnhancedInputLocalPlayerSubsystem* Subsystem;
		UHeadMountedDisplayFunctionLibrary::SetTrackingOrigin(TrackOrigin);
		
		IConsoleVariable* cSSPRT = IConsoleManager::Get().FindConsoleVariable(TEXT("xr.SecondaryScreenPercentage.HMDRenderTarget"));
		if(cSSPRT) { cSSPRT->Set(SecondaryScreenPercentage, EConsoleVariableFlags::ECVF_SetByCode); }

		bool bSubSystem = (Subsystem = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(GetLocalPlayer())) != nullptr;
		if (bSubSystem)
		{ 
			Subsystem->AddMappingContext(DefaultMappingContext, 0);
			Subsystem->AddMappingContext(HandsMappingContext, 0);
		}
		
		UXRLoadingScreenFunctionLibrary::HideLoadingScreen();
	}
}

void AVRPlayerController::Tick(float DeltaTime)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		//UCineCameraComponent* CineCamera = VRCamera->GetCineCamera();
	}

#if WITH_EDITORONLY_DATA
	_DEBUG_Draw_Limits();
#endif
}

void AVRPlayerController::OnPossess(APawn* InPawn)
{
	Super::OnPossess(InPawn);

	const bool bValid = InPawn->IsA<AVRCameraPawn>();
	if (!bValid)
	{
		UnPossess();
		const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
		if (bCamera) { Possess(VRCamera.Get()); }
	}

	VRCamera = Cast<AVRCameraPawn>(GetPawn());
	//ResolveRPC();
	InitForNewVRCamera();
}

//void AVRController::ResolveRPC(ACharacter* InRPC)
//{
//
//}

void AVRPlayerController::Move(const FInputActionValue& Value)
{
	FPredictProjectilePathParams TraceParms;
	FPredictProjectilePathResult TraceResult;

	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		MoveOrigin = VRCamera->RightAim->GetComponentLocation();
		MoveForward = VRCamera->RightAim->GetForwardVector();

		TraceParms = FPredictProjectilePathParams(TraceProjectileRadius, MoveOrigin, TraceLaunchVelocity * MoveForward, 2.0f);
		TraceParms.bTraceComplex = bTraceComplex;
		TraceParms.ObjectTypes = TraceObjectTypes;
		TraceParms.bTraceWithCollision = bTracePath;
		TraceParms.bTraceWithChannel = false;
		TraceParms.SimFrequency = 15.0f;

		bool bHit = UGameplayStatics::PredictProjectilePath(GetWorld(), TraceParms, TraceResult);
		if (bTraceUseNavMesh) { bHit = IsValidMoveLocation(TraceResult.HitResult, MoveTarget); }

		const bool bAttatch = bVRDynamic && bHit;//&& VRDynamicComponent != nullptr && VRDynamicLastComponent.IsValid();
		const bool bDetatch = bHit && VRDynamicLastComponent != nullptr && VRDynamicLastComponent.IsValid();
		if (bAttatch)
		{
			FAttachmentTransformRules Rule = FAttachmentTransformRules::KeepWorldTransform;
			Rule.bWeldSimulatedBodies = true;
			bVRDynamicAttached = VRCamera->AttachToComponent(VRDynamicComponent.Get(), Rule);
			if (bVRDynamicAttached) { VRDynamicLastComponent = VRDynamicComponent;  }
		}
		else if (bDetatch)
		{
			VRCamera->DetachFromActor(FDetachmentTransformRules::KeepWorldTransform);
			VRDynamicLastComponent = nullptr;
			bVRDynamicAttached = false;
		}

		const bool bUpdate = bHit != bMoveValid;
		if (bUpdate)
		{
			// Nothing Yet
		}

		bMoveValid = bHit;
	}
}

void AVRPlayerController::MoveStart(const FInputActionValue& Value)
{
	bTracePathShow = true;

}

void AVRPlayerController::MoveStop(const FInputActionValue& Value)
{
	bTracePathShow = false;

	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		if (bMoveValid)
		{
			bMoveValid = false;
			UCineCameraComponent* CineCamera = VRCamera->GetCineCamera();

			FVector Location = CineCamera->GetRelativeLocation();
			Location.Z = 0.0;
			
			FRotator Rotation = VRCamera->GetActorRotation();
			Rotation.Roll = Rotation.Yaw = 0.0;
			
			VRCamera->TeleportTo(MoveTarget - Rotation.RotateVector(Location), Rotation);
		}
	}
}

//void AVRController::Look(const FInputActionValue& Value)
//{
//
//}

void AVRPlayerController::Turn(const FInputActionValue& Value)
{
	float TurnValue = Value.Get<float>();
	SnapTurn(TurnValue > 0.0);
}

void AVRPlayerController::GrabLeft(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		VRCamera->LeftGrip;
	}
}

void AVRPlayerController::GrabLeftStop(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		VRCamera->LeftGrip;
	}
}

void AVRPlayerController::GrabRight(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		VRCamera->RightGrip;
	}
}

void AVRPlayerController::GrabRightStop(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		VRCamera->RightGrip;
	}
}

#if WITH_EDITORONLY_DATA
void AVRPlayerController::_DEBUG_Draw_Limits()
{
	if (!_DEBUG_Show_Draw_Limits) { return; }

	float Line_Length = 30;
	float Sphere_Size = 15;

	FColor Cur_Col = bMoveValid ? FColor(0, 255, 255) : FColor(255, 0, 0);
	if (bTracePathShow)
	{
		DrawDebugSphere(GetWorld(), MoveTarget, Sphere_Size, 16, Cur_Col, false, 0);
		DrawDebugSphere(GetWorld(), MoveOrigin, Sphere_Size, 16, Cur_Col, false, 0);
		DrawDebugLine(GetWorld(), MoveOrigin, MoveOrigin + MoveForward * Line_Length, Cur_Col, false, 0, 0u, _DEBUG_Line_Width);
	}
}
#endif