#pragma once

#include "MotionControllerComponent.h"
#include "VRNotificationsComponent.h"

#include "GameFramework/Character.h"
#include "CineCameraComponent.h"
#include "Logging/LogMacros.h"
#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "VRCameraPawn.generated.h"

class UCineCameraComponent;

DECLARE_LOG_CATEGORY_EXTERN(LogTemplateNoirPawn, Log, All);

UCLASS(config = Game)
class VACTVR_API AVRCameraPawn : public APawn
{
	GENERATED_BODY()

public:

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	USceneComponent* VROrigin;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UMotionControllerComponent* LeftGrip;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UMotionControllerComponent* RightGrip;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UMotionControllerComponent* LeftAim;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UMotionControllerComponent* RightAim;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UCineCameraComponent* CineCamera;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UVRNotificationsComponent* VRNotifications;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TWeakObjectPtr<ACharacter> RPC;

public:
	AVRCameraPawn();

	FORCEINLINE UCineCameraComponent* GetCineCamera() const { return CineCamera; }

};
