#include "InteractionStateComponent.h"

UInteractionStateComponent::UInteractionStateComponent()
{
	PrimaryComponentTick.bCanEverTick = true;

	State = FInteractionEvent::InvalidInteractionEvent;
	State.LastTime = FDateTime::UtcNow().ToUnixTimestamp();
	State.Name = GetFName();
	State.Id = 0;
	State.bEnabled = true;
}

void UInteractionStateComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (TargetComponent)
	{
		State.bInvalid = false;
		State.SecondsSince += DeltaTime;
		State.Component = TargetComponent.Get();
	}
}

