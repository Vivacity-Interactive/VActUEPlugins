#pragma once

#include "VActBeatsTypes.h"

#include "CoreMinimal.h"
#include "BeatVectorModern.generated.h"

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatVectorModern : public FBeatTime
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
	float Direction;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Approach;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float Focus;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Perception")
	float BAC;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Shame;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Resentment;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Anxious;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Excitement;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Confidence;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Confort;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Arousal;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Mindfull;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Empathy;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Sympathy;
};