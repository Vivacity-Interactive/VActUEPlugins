#pragma once

#include "VActBeatsTypes.h"
#include "BeatsProfile.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "BeatPool.generated.h"

UCLASS(editinlinenew, BlueprintType, Blueprintable)
class VACTBEATS_API UBeatPool : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (MultiLine = "true"))
	FString Description;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TObjectPtr<UBeatsProfile> Profile;

public:
	void Init(bool bClear = true);

};