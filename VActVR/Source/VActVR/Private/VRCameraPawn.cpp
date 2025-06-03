#include "VRCameraPawn.h"
/// = FName(TEXT("LeftGrip"))
// Sets default values
AVRCameraPawn::AVRCameraPawn()
{
 	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
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

