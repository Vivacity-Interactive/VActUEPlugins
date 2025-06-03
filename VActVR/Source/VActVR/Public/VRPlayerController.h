#pragma once

#include "InputAction.h"
#include "InputMappingContext.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "HeadMountedDisplayTypes.h"

#include "VRCameraPawn.h"
//#include "NiagaraComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "VRPlayerController.generated.h"


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

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	bool bTraceUseNavMesh;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	bool bTraceComplex;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	bool bTracePath;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	bool bMoveValid;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	bool bTracePathShow;
		
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR State")
	TEnumAsByte<EHMDTrackingOrigin::Type> TrackOrigin;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR State")
	TArray<TEnumAsByte<EObjectTypeQuery>> TraceObjectTypes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	float SecondaryScreenPercentage;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	float TraceProjectileRadius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	float TraceLaunchVelocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings")
	float SnapTurnDegree;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VR Settings", meta = (EditCondition = "bTraceUseNavMesh", EditConditionHides))
	float NavMeshCellHeight;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	FVector MoveForward;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	FVector MoveOrigin;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	FVector MoveTarget;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	TArray<FVector> TracePathPoints;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VR State")
	TArray<FVector> TracePath;

public:
	AVRPlayerController();

	//void InitForNewRPC();

	void InitForNewVRCamera();

	bool IsValidMoveLocation(const FHitResult& Hit, FVector& ProjectedLocation);

	void SnapTurn(bool bRight);

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
