#pragma once

#include "CameraHint.h"

#include "CoreMinimal.h"
#include "Engine/NavigationObjectBase.h"
#include "CameraSample.generated.h"

class ACameraHint;

struct FCameraSampleScoreCursor;

struct VACTCAMERA_API FCameraSampleScoreCursor
{
	float Score;

	ACameraSample* Sample;
	
	FCameraSampleScoreCursor* Next;
};

UCLASS()
class VACTCAMERA_API ACameraSample : public ANavigationObjectBase
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = NoirCameraSample)
	TSoftObjectPtr<ACameraHint> Hint;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = NoirCameraSample)
	float Damp;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = NoirCameraSample)
	FName CameraSampleTag;

public:
	ACameraSample(const FObjectInitializer& ObjectInitializer);
	
#if WITH_EDITORONLY_DATA
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = NoirCameraSample)
	bool bValidate;

	virtual void Validate() override;

	virtual void __DEBUG_Draw(float DeltaTime = 1.0f);
#endif
};
