#pragma once

#include "InteractionEventTypes.h"

#include "CoreMinimal.h"
#include "InteractionComponent.h"
#include "MonoInteractionComponent.generated.h"

UCLASS(Blueprintable, ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VACTVR_API UMonoInteractionComponent : public UInteractionComponent
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	bool OnMonoInteraction(
		const FInteractionEvent& Event,
		const EInteractionEvent EventType,
		float DeltaTime
	);

};
