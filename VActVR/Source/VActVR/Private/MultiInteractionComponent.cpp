#include "MultiInteractionComponent.h"

UMultiInteractionComponent::UMultiInteractionComponent()
	: MinActivationEvents(-1)
	, MaxActivationEvents(-1)
{
}

bool UMultiInteractionComponent::OnMultiInteraction_Implementation(
		const TArray<FInteractionEvent>& Event,
		const EInteractionEvent EventType,
		float DeltaTime
	)
{
	return true;
}