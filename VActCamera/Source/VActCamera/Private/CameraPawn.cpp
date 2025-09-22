#include "CameraPawn.h"

ACameraPawn::ACameraPawn()
{
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	if (!DefaultRootComponent)
	{
		DefaultRootComponent = CreateDefaultSubobject<USphereComponent>(TEXT("DefaultRootComponent"));
		RootComponent = DefaultRootComponent;
	}
	
	if (!CineCameraComponent)
	{
		CineCameraComponent = CreateDefaultSubobject<UCineCameraComponent>(TEXT("CineCameraComponent"));
		CineCameraComponent->SetupAttachment(RootComponent);
		CineCameraComponent->bUsePawnControlRotation = false;
	}
}

UCineCameraComponent* ACameraPawn::GetCineCameraComponent()
{
	return CineCameraComponent;
}

void ACameraPawn::AssignCameraVector(const FCameraVector& Vector)
{
	if (CineCameraComponent)
	{
		FCameraLensSettings& Lens = CineCameraComponent->LensSettings;
		FCameraFocusSettings& Focus = CineCameraComponent->FocusSettings;
		FPostProcessSettings& PostProcessing = CineCameraComponent->PostProcessSettings;

		Lens.MaxFocalLength = Vector.FocalLengthMax;
		Lens.MinFocalLength = Vector.FocalLengthMin;
		
		PostProcessing.AutoExposureBias = Vector.Exposure;
		PostProcessing.DepthOfFieldSensorWidth = Vector.SensorWidth;
		CineCameraComponent->SetCurrentAperture(Vector.Aperture);

		const bool bAutoTrack = Focus.TrackingFocusSettings.ActorToTrack != nullptr && Focus.TrackingFocusSettings.ActorToTrack.IsValid();
		if (!bAutoTrack) { PostProcessing.DepthOfFieldFocalDistance = Vector.FocalDistance; }
		
		//SetActorLocation(Vector.Location);
		//SetActorRotation(Vector.Rotation);
	}
}

void ACameraPawn::ExtractCameraVector(FCameraVector& Vector)
{
	if (CineCameraComponent)
	{
		const FCameraLensSettings& Lens = CineCameraComponent->LensSettings;
		const FCameraFocusSettings& Focus = CineCameraComponent->FocusSettings;
		const FPostProcessSettings& PostProcessing = CineCameraComponent->PostProcessSettings;

		Vector.FocalLengthMax = Lens.MaxFocalLength;
		Vector.FocalLengthMin = Lens.MinFocalLength;
		
		Vector.Exposure = PostProcessing.AutoExposureBias;
		Vector.SensorWidth = PostProcessing.DepthOfFieldSensorWidth;
		Vector.Aperture = CineCameraComponent->CurrentAperture;
		Vector.FocalDistance = PostProcessing.DepthOfFieldFocalDistance;

		Vector.Location = GetActorLocation();
		Vector.Rotation = GetActorRotation();
	}
}

void ACameraPawn::AssignCameraOperatorVector(const FCameraOperatorVector& Vector)
{
	if (CineCameraComponent)
	{
		CineCameraComponent->SetCurrentFocalLength(Vector.FocalLengt);
		SetActorLocation(Vector.Follow);
		SetActorRotation(Vector.Track);
	}
}

void ACameraPawn::ExtractCameraOperatorVector(FCameraOperatorVector& Vector)
{
	if (CineCameraComponent)
	{
		Vector.FocalLengt = CineCameraComponent->CurrentFocalLength;
		Vector.Follow = GetActorLocation();
		Vector.Track = GetActorRotation();
	}
}