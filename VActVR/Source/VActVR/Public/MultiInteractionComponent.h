#pragma once

#include "InteractionEventTypes.h"

#include "CoreMinimal.h"
#include "InteractionComponent.h"
#include "MultiInteractionComponent.generated.h"

UCLASS(Blueprintable, ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VACTVR_API UMultiInteractionComponent : public UInteractionComponent
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	int32 MinActivationEvents;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	int32 MaxActivationEvents;

public:
	UMultiInteractionComponent();

public:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	bool OnMultiInteraction(
		const TArray<FInteractionEvent>& Event,
		const EInteractionEvent EventType,
		float DeltaTime
	);
	
};
