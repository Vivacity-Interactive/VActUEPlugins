#pragma once

#include "GameFramework/Actor.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "BeatAction.generated.h"

class AActor;
class AController;

struct FBeatContext;

UCLASS(editinlinenew, BlueprintType, Blueprintable)
class VACTBEATS_API UBeatAction : public UObject
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Execute(
		float DeltaTime, 
		AActor* Actor, 
		AActor* Context,
		const FBeatContext& BeatContext,
		AActor* Owner = nullptr
	);
};