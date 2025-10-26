#pragma once

#define _VACT_SIZE_T SIZE_T

#include "VActSTTTypes.h"

#include "CoreMinimal.h"
#include "VActSTT.generated.h"

struct FSTTModel;

USTRUCT()
struct FVActSTT
{
	GENERATED_BODY()

	static void _Unsafe_Create(FSTTModel& Into);
	
	static void _Unsafe_Destroy(FSTTModel& Tensor, bool bData = true);

	static FSTTModel CreateModel();
	
	static void Destroy(FSTTModel& Tensor, bool bData = true);

};
