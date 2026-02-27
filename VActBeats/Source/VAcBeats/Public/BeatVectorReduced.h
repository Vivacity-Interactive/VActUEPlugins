#pragma once

#include "VActBeatsTypes.h"

#include "CoreMinimal.h"
#include "BeatVectorReduced.generated.h"

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatVectorReduced : public FBeatTime
{
	static const TArray<FName> FeatureNames;

	static const TMap<FName, int32> MapFeatureNames;

	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Interrupt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Distance;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Facing;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Focus;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Comfort;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Secure;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Trusting;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Awareness;
};