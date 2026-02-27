#pragma once

#include "VActBeatsTypes.h"

#include "CoreMinimal.h"
#include "BeatVectorClassic.generated.h"

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatVectorClassic : public FBeatTime
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
	float Anger;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Contempt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Disgust;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Enjoyment;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Fear;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Sadness;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Emotion")
	float Surprise;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Mindfull;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Empathy;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Reflection")
	float Sympathy;
};