#include "DualInteractionComponent.h"

UDualInteractionComponent::UDualInteractionComponent()
	: bDuelActivationEvents(false)
{
}

bool UDualInteractionComponent::OnDualInteraction_Implementation(
		const FInteractionEvent& PrimaryEvent,
		const FInteractionEvent& SecondaryEvent,
		const EInteractionEvent EventType,
		float DeltaTime
	)
{
	return true;
}