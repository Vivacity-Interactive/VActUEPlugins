#include "InteractionEventTypes.h"

const FInteractionEvent FInteractionEvent::InvalidInteractionEvent = FInteractionEvent();

FInteractionEvent::FInteractionEvent()
	: bInvalid(true)
	, bEnabled(false)
	, bCause(false)
	, bActive(false)
	, LastTime(0)
	, Name(NAME_None)
	, Slot(NAME_None)
	, Id(-1)
	, SecondsSince(0.0f)
	, Component(nullptr)
	, Context(nullptr)
{
}

bool FInteractionGrip::IsValid() const
{
	return Socket.Get() || Collider.Get();
}

USceneComponent* FInteractionGrip::GetSocket() const
{
	return Socket ? Socket.Get() : Collider.Get();
}