#include "MonoInteractionComponent.h"

bool UMonoInteractionComponent::OnMonoInteraction_Implementation(
		const FInteractionEvent& Event,
		const EInteractionEvent EventType,
		float DeltaTime
	)
{
	return true;
}