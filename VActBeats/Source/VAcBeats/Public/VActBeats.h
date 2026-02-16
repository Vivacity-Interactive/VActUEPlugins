#pragma once

#include "VActBeatsTypes.h"
#include "CoreMinimal.h"

struct VACTBEATS_API FVActBeats
{
	const static TMap<FName, int32> IdBeatVector;

	const static TMap<FName, int32> IdBeatVectorMinimal;
	
	const static TArray<FName> NameBeatVector;

	const static TArray<FName> NameBeatVectorMinimal;

	FVActBeats() = delete;

};
