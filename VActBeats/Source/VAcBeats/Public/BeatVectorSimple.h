#pragma once

#include "VActBeatsTypes.h"

#include "CoreMinimal.h"
#include "BeatVectorSimple.generated.h"

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatVectorSimple : public FBeatTime
{
	const static TArray<FName> FeatureNames;

	const static TMap<FName, int32> MapFeatureNames;

	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Distance;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Focus;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Friendly;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Safe;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Aware;
};