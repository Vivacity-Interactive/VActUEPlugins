#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "VActToonRayGen.h"
#include "VActToonRunner.generated.h"

UCLASS()
class VACTTOON_API AVActToonRunner : public AActor
{
	GENERATED_BODY()

public:
	AVActToonRunner();
	FVActToonRayGen Toon;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = ShaderDemo)
	class UTextureRenderTarget2D* RenderTarget = nullptr;

protected:
	virtual void BeginPlay() override;
	void UpdateToonParameters();

	float TranscurredTime; ///< allows us to add a delay on BeginPlay() 
	bool Initialized;

public:
	virtual void Tick(float DeltaTime) override;
	virtual void BeginDestroy() override;
};