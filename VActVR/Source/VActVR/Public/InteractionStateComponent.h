#pragma once

#include "InteractionEventTypes.h"

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InteractionStateComponent.generated.h"


UCLASS(Blueprintable, ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VACTVR_API UInteractionStateComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (AllowComponentRef));
	TObjectPtr<UPrimitiveComponent> TargetComponent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	FInteractionEvent State;

public:
	UInteractionStateComponent();

public:
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

};
