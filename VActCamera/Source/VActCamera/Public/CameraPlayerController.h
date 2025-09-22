#pragma once

#include "InputMappingContext.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputActionValue.h"

#include "CameraPawn.h"
#include "CameraSample.h"
#include "VActCamera.h"

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "CameraPlayerController.generated.h"

class APawn;
class ACameraPawn;
class ACameraSample;
class UInputMappingContext;
class UInputAction;

struct FInputActionValue;
struct FCameraSettings;

USTRUCT(BlueprintType)
struct VACTCAMERA_API FCameraPlayerInputActions
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	UInputMappingContext* DefaultMappingContext;

	// General Actions

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "General Actions", meta = (AllowPrivateAccess = "true"))
	UInputAction* MoveAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "General Actions", meta = (AllowPrivateAccess = "true"))
	UInputAction* LookAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "General Actions", meta = (AllowPrivateAccess = "true"))
	UInputAction* ZoomAction;

};


UCLASS()
class VACTCAMERA_API ACameraPlayerController : public APlayerController
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input")
	uint8 bInvertY : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	uint8 bUseCameraHints : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	uint8 bUseCameraSampleRadius : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	uint8 bSetCurrentTargetBeginPlay : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	uint8 bCameraEnableCollision : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	uint8 bUseCameraTrackOffset : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RPC")
	uint8 bRPCOrientRotationToMovement : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RPC")
	FName RPCTagName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FName FocusSocketName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FName FollowSocketName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FName TrackSocketName;

	UPROPERTY(Interp, EditAnywhere, BlueprintReadWrite, Category = "RPC")
	TSoftObjectPtr<APawn> RPC;

	UPROPERTY(Interp, EditAnywhere, BlueprintReadWrite, Category = "Camera")
	TSoftObjectPtr<ACameraPawn> CameraPawn;

	UPROPERTY(Interp, EditAnywhere, BlueprintReadWrite, Category = "Camera|Hint")
	TSoftObjectPtr<ACameraSample> CameraSample;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input")
	FCameraPlayerInputActions InputActions;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Hint")
	float CameraHintRadius;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera|Hint")
	float CameraHintDamp;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Hint")
	float CameraHintDampTarget;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Hint")
	float CameraHintDampSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Camera|Hint")
	float CameraHintProgress;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	float MaxOperatorDampVelocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input")
	float ZoomSensitivityScale;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Input")
	float LookSensitivityScale;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Hint")
	FVector HintLocationAlpha;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Hint")
	FVector HintRotationAlpha;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	FCameraVector CameraCurrent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FCameraVector CameraTarget;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	FCameraOperatorVector OperatorCurrent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	FCameraOperatorVector OperatorVelocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FCameraOperatorVector OperatorTarget;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FCameraOperatorVector OperatorDamp;

public:
	ACameraPlayerController();

	void InitForNewRPC();

	void InitForNewNoirCamera();
	
	FVector ComputeCursorLocation(APawn* InRPC = nullptr) const;

protected:
	virtual void SetupInputComponent() override;

	virtual void BeginPlay() override;

	virtual void Tick(float DeltaTime) override;

	virtual void OnPossess(APawn* InPawn) override;

	virtual void ResolveRPC(APawn* InRPC = nullptr);

protected:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Input")
	void Move(const FInputActionValue& Value);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Input")
	void Look(const FInputActionValue& Value);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Input")
	void Zoom(const FInputActionValue& Value);

};
