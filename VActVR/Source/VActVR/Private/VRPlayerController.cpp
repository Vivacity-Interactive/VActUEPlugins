#include "VRPlayerController.h"

#include "VActVR.h"

#include "Engine/EngineTypes.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputActionValue.h"
#include "XRLoadingScreenFunctionLibrary.h"
#include "HAL/IConsoleManager.h"
#include "Kismet/GameplayStatics.h"
#include "NavigationSystem.h"

FVRInteractionContextEvent::FVRInteractionContextEvent()
	: Context(nullptr)
	, Instigator(nullptr)
	, Controller(nullptr)
{

}

FVRInteractionContextEvent::FVRInteractionContextEvent(
	UPrimitiveComponent* InContext,
	UMotionControllerComponent* InInstigator,
	AVRPlayerController* InController
)
	: Context(InContext)
	, Instigator(InInstigator)
	, Controller(InController)
{

}

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
	bEnableLeftGrab = true;
	bEnableRightGrab = true;
	bParentOnInteaction = true;
	bOnEmptyHandOnly = true;
	bInteactionTraceComplex = false;
	bEnableInteractionCallback = false;
	//bImpulseOnRelease = true;
	//bEnableTeleport = true;
	PhysicsTrackFlagName = FName("_VR_IsSimulatingPhysics");
	SnapTurnDegree = -45.0f;
	NavMeshCellHeight = 8.0f;
	InteractionRadius = 10.0f;

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

		// TODO: Fix resets orientation bug when snap turning
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
		if (cSSPRT) { cSSPRT->Set(SecondaryScreenPercentage, EConsoleVariableFlags::ECVF_SetByCode); }

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

		// Maybe move this to FVActVR
		TraceParms = FPredictProjectilePathParams(TraceProjectileRadius, MoveOrigin, TraceLaunchVelocity * MoveForward, 2.0f);
		TraceParms.bTraceComplex = bTraceComplex;
		TraceParms.ObjectTypes = TraceObjectTypes;
		TraceParms.bTraceWithCollision = bTracePath;
		TraceParms.bTraceWithChannel = false;
		TraceParms.SimFrequency = 15.0f;

		bool bHit = UGameplayStatics::PredictProjectilePath(GetWorld(), TraceParms, TraceResult);
		if (bTraceUseNavMesh) { bHit = IsValidMoveLocation(TraceResult.HitResult, MoveTarget); }

		bool bTrachePath = TraceResult.PathData.IsEmpty();
		if (bTrachePath) { TracePathPoints = MoveTemp(TraceResult.PathData); }

		const bool bAttatch = bVRDynamic && bHit;//&& VRDynamicComponent != nullptr && VRDynamicLastComponent.IsValid();
		const bool bDetatch = bHit && VRDynamicLastComponent != nullptr && VRDynamicLastComponent.IsValid();
		if (bAttatch)
		{
			FAttachmentTransformRules Rule = FAttachmentTransformRules::KeepWorldTransform;
			Rule.bWeldSimulatedBodies = true;
			bVRDynamicAttached = VRCamera->AttachToComponent(VRDynamicComponent.Get(), Rule);
			if (bVRDynamicAttached) { VRDynamicLastComponent = VRDynamicComponent; }
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
		bool bTryGrab = bEnableLeftGrab 
			&& (!bOnEmptyHandOnly || !LeftInteractingComponent.IsValid() );

		if (bTryGrab)
		{
			FHitResult Hit;
			FVector Origin = VRCamera->LeftGrip->GetComponentLocation();
			const bool bHit = FVActVR::TraceSphere(
				this, Origin, Origin, InteractionRadius,
				InteractionTraceObjectTypes, Hit,
				FVActVR::EmptyActorArray, true, bInteactionTraceComplex
			);

			if (bHit)
			{
				if (bEnableInteractionCallback)
				{
					bTryGrab &= OnInteraction(
						Value, 0,
						FVRInteractionContextEvent(Hit.GetComponent(), VRCamera->LeftGrip, this),
						EInteractionEvent::OnTriggered
					);

					if (!bTryGrab) { return; }
				}

				const bool bUnparent = bParentOnInteaction 
					&& LeftInteractingComponent.IsValid()
					&& LeftInteractingComponent->GetAttachParent() == VRCamera->LeftGrip;

				if (bUnparent)
				{
					LeftInteractingComponent->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
					LeftInteractingComponent->SetSimulatePhysics(LeftInteractingComponent->ComponentHasTag(PhysicsTrackFlagName));
				}

				LeftInteractingComponent = nullptr;
				LeftInteractingComponent = Hit.GetComponent();

				// TODO: this is a fix, but may also lead to physics simulation consuming towards true when inteacted
				const bool bFlagPhysics = LeftInteractingComponent->IsSimulatingPhysics();
				if (bFlagPhysics)
				{
					LeftInteractingComponent->ComponentTags.AddUnique(PhysicsTrackFlagName);
				}

				const bool bParent = bParentOnInteaction && LeftInteractingComponent.IsValid();
				if (bParent)
				{
					FAttachmentTransformRules Rule = FAttachmentTransformRules::KeepWorldTransform;
					Rule.bWeldSimulatedBodies = bWieldSimulatingBodies;
					LeftInteractingComponent->AttachToComponent(VRCamera->LeftGrip, Rule);
				}
			}
		}
	}
}

void AVRPlayerController::GrabLeftStop(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		bool bTryRelease = bEnableLeftGrab;

		if (bEnableInteractionCallback)
		{
			bTryRelease &= OnInteraction(
				Value, 0,
				FVRInteractionContextEvent(LeftInteractingComponent.Get(), VRCamera->LeftGrip, this),
				EInteractionEvent::OnCompleted
			);
		}

		if (bTryRelease)
		{
			bool bUnparent = bParentOnInteaction
				&& LeftInteractingComponent.IsValid()
				&& LeftInteractingComponent->GetAttachParent() == VRCamera->LeftGrip;

			if (bUnparent)
			{
				LeftInteractingComponent->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
				LeftInteractingComponent->SetSimulatePhysics(LeftInteractingComponent->ComponentHasTag(PhysicsTrackFlagName));
			}
			LeftInteractingComponent = nullptr;
		}
	}
}

void AVRPlayerController::GrabRight(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		bool bTryGrab = bEnableRightGrab
			&& (!bOnEmptyHandOnly || !RightInteractingComponent.IsValid());

		if (bTryGrab)
		{
			FHitResult Hit;
			FVector Origin = VRCamera->RightGrip->GetComponentLocation();
			const bool bHit = FVActVR::TraceSphere(
				this, Origin, Origin, InteractionRadius,
				InteractionTraceObjectTypes, Hit,
				FVActVR::EmptyActorArray, true, bInteactionTraceComplex
			);

			if (bHit)
			{
				if (bEnableInteractionCallback)
				{
					bTryGrab &= OnInteraction(
						Value, 1,
						FVRInteractionContextEvent(Hit.GetComponent(), VRCamera->RightGrip, this),
						EInteractionEvent::OnTriggered
					);

					if (!bTryGrab) { return; }
				}

				const bool bUnparent = bParentOnInteaction 
					&& RightInteractingComponent.IsValid()
					&& RightInteractingComponent->GetAttachParent() == VRCamera->RightGrip;

				if (bUnparent)
				{
					RightInteractingComponent->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
					RightInteractingComponent->SetSimulatePhysics(RightInteractingComponent->ComponentHasTag(PhysicsTrackFlagName));
				}
				
				RightInteractingComponent = nullptr;
				RightInteractingComponent = Hit.GetComponent();
				
				// TODO: this is a fix, but may also lead to physics simulation consuming towards true when inteacted
				const bool bFlagPhysics = RightInteractingComponent->IsSimulatingPhysics();				
				if (bFlagPhysics)
				{
					RightInteractingComponent->ComponentTags.AddUnique(PhysicsTrackFlagName);
				}
				
				const bool bParent = bParentOnInteaction && RightInteractingComponent.IsValid();
				if (bParent)
				{
					FAttachmentTransformRules Rule = FAttachmentTransformRules::KeepWorldTransform;
					Rule.bWeldSimulatedBodies = bWieldSimulatingBodies;
					RightInteractingComponent->AttachToComponent(VRCamera->RightGrip, Rule);
				}
			}
		}
	}
}

void AVRPlayerController::GrabRightStop(const FInputActionValue& Value)
{
	const bool bCamera = VRCamera != nullptr && VRCamera.IsValid();
	if (bCamera)
	{
		bool bTryRelease = bEnableRightGrab;
		
		if (bEnableInteractionCallback)
		{
			bTryRelease &= OnInteraction(
				Value, 1,
				FVRInteractionContextEvent(RightInteractingComponent.Get(), VRCamera->RightGrip, this),
				EInteractionEvent::OnCompleted
			);
		}

		if (bTryRelease)
		{
			const bool bUnparent = bParentOnInteaction
				&& RightInteractingComponent.IsValid()
				&& RightInteractingComponent->GetAttachParent() == VRCamera->RightGrip;

			if (bUnparent)
			{
				RightInteractingComponent->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
				RightInteractingComponent->SetSimulatePhysics(RightInteractingComponent->ComponentHasTag(PhysicsTrackFlagName));
			}
			RightInteractingComponent = nullptr;
		}
	}
}

FVRInteractionContextEvent AVRPlayerController::GetLeftInteractionContextEvent()
{
	return FVRInteractionContextEvent(LeftInteractingComponent.Get(), VRCamera.IsValid() ? VRCamera->LeftGrip : nullptr, this);
}

FVRInteractionContextEvent AVRPlayerController::GetRightInteractionContextEvent()
{
	return FVRInteractionContextEvent(RightInteractingComponent.Get(), VRCamera.IsValid() ? VRCamera->RightGrip : nullptr, this);
}

bool AVRPlayerController::OnInteraction_Implementation(
		const FInputActionValue& Value, int32 Id,
		const FVRInteractionContextEvent& ContextEvent,
		EInteractionEvent EventMode,
		float DeltaTime
	)
{
	return true;
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