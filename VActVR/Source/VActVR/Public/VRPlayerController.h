#pragma once

#include "InputAction.h"
#include "InputMappingContext.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "HeadMountedDisplayTypes.h"

#include "InteractionEventTypes.h"

#include "Kismet/GameplayStaticsTypes.h"

#include "VRCameraPawn.h"
//#include "NiagaraComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "VRPlayerController.generated.h"

USTRUCT(BlueprintType)
struct VACTVR_API FVRInteractionContextEvent
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly)
	UPrimitiveComponent* Context;

	UPROPERTY(BlueprintReadOnly)
	UMotionControllerComponent* Instigator;

	UPROPERTY(BlueprintReadOnly)
	AVRPlayerController* Controller;

	FVRInteractionContextEvent();

	FVRInteractionContextEvent(
		UPrimitiveComponent* InContext,
		UMotionControllerComponent* InInstigator,
		AVRPlayerController* InController
	);
};


UCLASS()
class VACTVR_API AVRPlayerController : public APlayerController
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	UInputMappingContext* DefaultMappingContext;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	UInputMappingContext* HandsMappingContext;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	UInputAction* MoveAction;

	//UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	//UInputAction* LookAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	UInputAction* TurnAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	UInputAction* GrabLeftAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta = (AllowPrivateAccess = "true"))
	UInputAction* GrabRightAction;

public:
	UPROPERTY(Interp, EditAnywhere, BlueprintReadWrite)
	TSoftObjectPtr<AVRCameraPawn> VRCamera;

	//UPROPERTY(Interp, EditAnywhere, BlueprintReadWrite)
	//TSoftObjectPtr<ACharacter> RPC;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite)
	//FName RPCTagName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	uint8 bTraceUseNavMesh : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	uint8 bTraceComplex : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	uint8 bTracePath : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	uint8 bMoveValid : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	uint8 bTracePathShow : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	uint8 bVRDynamic : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	uint8 bVRDynamicAttached : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bEnableLeftGrab : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bEnableRightGrab : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bParentOnInteaction : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bOnEmptyHandOnly : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bWieldSimulatingBodies : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bInteactionTraceComplex : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	uint8 bEnableInteractionCallback : 1;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	//uint8 bImpulseOnRelease : 1;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	//uint8 bEnableTeleport : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	FName PhysicsTrackFlagName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	TEnumAsByte<EHMDTrackingOrigin::Type> TrackOrigin;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	TArray<TEnumAsByte<EObjectTypeQuery>> TraceObjectTypes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	TEnumAsByte<ECollisionChannel> DynamicObjectChannel;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	TArray<TEnumAsByte<EObjectTypeQuery>> InteractionTraceObjectTypes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	float SecondaryScreenPercentage;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	float TraceProjectileRadius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	float TraceLaunchVelocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings")
	float SnapTurnDegree;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Settings", meta = (EditCondition = "bTraceUseNavMesh", EditConditionHides))
	float NavMeshCellHeight;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR|Interaction")
	float InteractionRadius;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	FVector MoveForward;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	FVector MoveOrigin;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	FVector MoveTarget;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	TArray<FPredictProjectilePathPointData> TracePathPoints;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "VR|State")
	TArray<FVector> TracePath;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	TWeakObjectPtr<UPrimitiveComponent> VRDynamicComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	TWeakObjectPtr<UPrimitiveComponent> VRDynamicLastComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	TWeakObjectPtr<UPrimitiveComponent> LeftInteractingComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR|State")
	TWeakObjectPtr<UPrimitiveComponent> RightInteractingComponent;

public:
	AVRPlayerController();

	//void InitForNewRPC();

	void InitForNewVRCamera();

	bool IsValidMoveLocation(const FHitResult& Hit, FVector& ProjectedLocation);

	void SnapTurn(bool bRight);

public:
	UFUNCTION(BlueprintCallable)
	FVRInteractionContextEvent GetLeftInteractionContextEvent();

	UFUNCTION(BlueprintCallable)
	FVRInteractionContextEvent GetRightInteractionContextEvent();

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	bool OnInteraction(
		const FInputActionValue& Value, int32 Id,
		const FVRInteractionContextEvent& ContextEvent,
		EInteractionEvent EventMode,
		float DeltaTime = 0.0f
	);

protected:
	virtual void SetupInputComponent() override;

	virtual void BeginPlay() override;

	virtual void Tick(float DeltaTime) override;

	virtual void OnPossess(APawn* InPawn) override;

	//virtual void ResolveRPC(ACharacter* InRPC = nullptr);

	void Move(const FInputActionValue& Value);

	void MoveStart(const FInputActionValue& Value);

	void MoveStop(const FInputActionValue& Value);

	//void Look(const FInputActionValue& Value);

	void Turn(const FInputActionValue& Value);

	void GrabLeft(const FInputActionValue& Value);

	void GrabLeftStop(const FInputActionValue& Value);

	void GrabRight(const FInputActionValue& Value);

	void GrabRightStop(const FInputActionValue& Value);

#if WITH_EDITORONLY_DATA
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = _Debug)
	bool _DEBUG_Show_Draw_Limits;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = _Debug)
	float _DEBUG_Line_Width;

	void _DEBUG_Draw_Limits();
#endif
};
