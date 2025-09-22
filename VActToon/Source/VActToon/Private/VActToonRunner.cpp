#include "VActToonRunner.h"
#include "Engine/World.h"
#include "SceneInterface.h"
#include "VActToonRayGen.h"
#include "../Private/ScenePrivate.h"

AVActToonRunner::AVActToonRunner()
{
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AVActToonRunner::BeginPlay()
{
	Super::BeginPlay();
	Toon = FVActToonRayGen();
	Initialized = false;

	if (RenderTarget != nullptr)
		UpdateToonParameters();
}

void AVActToonRunner::UpdateToonParameters()
{
	FVActToonRayGenParameters Parameters;
	Parameters.Scene = &GetWorld()->Scene->GetRenderScene()->RayTracingScene;
	Parameters.CachedRenderTargetSize = FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY);
	Parameters.RenderTarget = RenderTarget;
	Toon.UpdateParameters(Parameters);
}

// Called every frame
void AVActToonRunner::Tick(float DeltaTime)
{
	TranscurredTime += DeltaTime;
	Super::Tick(DeltaTime);

	// we want a slight delay before we start, otherwise some resources such as the accelerated structure will not be ready
	if (RenderTarget != nullptr && TranscurredTime > 1.0f)
	{
		UpdateToonParameters();

		if (!Initialized)
		{
			Toon.BeginRendering();
			Initialized = true;
		}
	}
}
void AVActToonRunner::BeginDestroy()
{
	Super::BeginDestroy();
	Toon.EndRendering();
}