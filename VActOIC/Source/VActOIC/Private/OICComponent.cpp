#include "OICComponent.h"

UOICComponent::UOICComponent()
{
	bIsEditorOnly = true;
	Profile.bClear = true;
}

void UOICComponent::OnRegister()
{
	Super::OnRegister();
	OICManager = AOICManagerActor::GetInstance(GetWorld());
	Profile.Parent = GetOwner();
	Profile.bClear = true;
}

void UOICComponent::UpdateProfile()
{
	const bool bManager = (OICManager != nullptr && OICManager.IsValid()) 
		|| (OICManager = AOICManagerActor::GetInstance(GetWorld())) != nullptr;

	Profile.Parent = GetOwner();
	if (bManager) { OICManager->UpdateProfile(GetWorld(), Profile); }
}

void UOICComponent::OnComponentDestroyed(bool bDestroyingHierarchy)
{
	Super::OnComponentDestroyed(bDestroyingHierarchy);
	const bool bManager = OICManager != nullptr && OICManager.IsValid();
	Profile.bClear = true;
	if (bManager) { OICManager->ClearProfile(Profile); }
}