#pragma once

#include "InteractionEventTypes.h"

#include "CoreMinimal.h"
#include "InteractionComponent.h"
#include "DualInteractionComponent.generated.h"

UCLASS(Blueprintable, ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VACTVR_API UDualInteractionComponent : public UInteractionComponent
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	uint8 bDuelActivationEvents : 1;

public:
	UDualInteractionComponent();

public:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	bool OnDualInteraction(
		const FInteractionEvent& PrimaryEvent,
		const FInteractionEvent& SecondaryEvent,
		const EInteractionEvent EventType,
		float DeltaTime
	);
};