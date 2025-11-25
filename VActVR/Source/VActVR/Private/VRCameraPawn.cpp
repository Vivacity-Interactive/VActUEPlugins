#include "VRCameraPawn.h"

AVRCameraPawn::AVRCameraPawn()
{
	PrimaryActorTick.bCanEverTick = true;

	VROrigin = CreateDefaultSubobject<USceneComponent>(TEXT("VROrigin"));
	SetRootComponent(VROrigin);

	CineCamera = CreateDefaultSubobject<UCineCameraComponent>(TEXT("CineCamera"));
	CineCamera->SetupAttachment(VROrigin);

	LeftGrip = CreateDefaultSubobject<UMotionControllerComponent>(TEXT("LeftGrip"));
	LeftGrip->SetTrackingMotionSource(FName("LeftGrip"));
	LeftGrip->SetupAttachment(VROrigin);

	RightGrip = CreateDefaultSubobject<UMotionControllerComponent>(TEXT("RightGrip"));
	RightGrip->SetTrackingMotionSource(FName("RightGrip"));
	RightGrip->SetupAttachment(VROrigin);

	LeftAim = CreateDefaultSubobject<UMotionControllerComponent>(TEXT("LeftAim"));
	LeftAim->SetTrackingMotionSource(FName("LeftAim"));
	LeftAim->SetupAttachment(VROrigin);

	RightAim = CreateDefaultSubobject<UMotionControllerComponent>(TEXT("RightAim"));
	RightAim->SetTrackingMotionSource(FName("RightAim"));
	RightAim->SetupAttachment(VROrigin);

	VRNotifications = CreateDefaultSubobject<UVRNotificationsComponent>(TEXT("VRNotifications"));

}

