#pragma once

#include "CoreMinimal.h"
#include "VACTSTTTypes.generated.h"

UENUM()
enum class ESTTModel
{
    TinyEnglish_Q8,
    Tiny_Q8,
    BaseEnglish_Q8,
    Base_Q8,
    SmallEnglish_Q8,
    Small_Q8,
    Turbo_Q8
};

USTRUCT()
struct VACTSTT_API FSTTContext
{
    GENERATED_BODY()
    


    void* _Context;
};

USTRUCT()
struct VACTSTT_API FSTTModel
{
    GENERATED_BODY()
    
};