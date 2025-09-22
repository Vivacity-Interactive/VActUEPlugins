#pragma once

#include "Components/SphereComponent.h"
#include "CineCameraComponent.h"
#include "VActCamera.h"

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "CameraPawn.generated.h"

UCLASS()
class VACTCAMERA_API ACameraPawn : public APawn
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	USphereComponent* DefaultRootComponent;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	UCineCameraComponent* CineCameraComponent;

public:
	ACameraPawn();

	UCineCameraComponent* GetCineCameraComponent();

	void AssignCameraVector(const FCameraVector& Vector);

	void ExtractCameraVector(FCameraVector& Vector);

	void AssignCameraOperatorVector(const FCameraOperatorVector& Vector);

	void ExtractCameraOperatorVector(FCameraOperatorVector& Vector);

};
