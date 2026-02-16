#pragma once

#include "BeatAction.h"

#include "Animation/AnimationAsset.h"
#include "Animation/AnimInstance.h"
#include "Sound/DialogueWave.h"
#include "Sound/SoundWave.h"

#include "CoreMinimal.h"
#include "BeatContext.generated.h"

class UBeatAction;
class UAnimInstance;
class UAnimationAsset;
class UDialogueWave;

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatContext
{
    GENERATED_BODY()
    
#if WITH_EDITORONLY_DATA
    UPROPERTY(EditAnywhere, Category = "Context", meta = (MultiLine = "true"))
	FString Notes;
#endif
    UPROPERTY(EditAnywhere, Category = "Context")
    FString Title;

    UPROPERTY(EditAnywhere, Category = "Context")
    FName Name;

    UPROPERTY(EditAnywhere, Category = "Context|Action")
	TSubclassOf<UBeatAction> ActionClass;

	UPROPERTY(EditAnywhere, Category = "Context|Action")
	TObjectPtr<UBeatAction> Action;

    UPROPERTY(EditAnywhere, Category = "Context|Dialogue")
	TObjectPtr<UDialogueWave> DialogueWave;

    UPROPERTY(EditAnywhere, Category = "Context|Dialogue")
    FString Dialogue;

    UPROPERTY(EditAnywhere, Category = "Context|Dialogue", meta = (DisallowedClasses = "/Script/MetasoundEngine.MetaSoundSource, /Script/Engine.SoundSourceBus"))
    TObjectPtr<USoundWave> SoundWave;

    UPROPERTY(EditAnywhere, Category = "Context|Animation")
	TSubclassOf<UAnimInstance> AnimInstanceClass;

    UPROPERTY(EditAnywhere, Category = "Context|Animation")
    TObjectPtr<UAnimationAsset> AnimAsset;
};