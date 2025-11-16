#include "APICallback.h"
#include "APIInstance.h"

#include "EngineUtils.h"

UObject* UAPICallback::GetContextObject() const
{
	return ContextObject.IsValid() ? ContextObject.Get() : nullptr;
}

bool UAPICallback::OnDataIn(
		const FHttpServerRequest& Request,
		const FAPIEntry& SelfEntry,
		UAPIRoute* Parent,
		UAPIInstance* Instance,
		FGuid& UserId
	)
{
	return false;
}

bool UAPICallback::OnDataOut(
		TUniquePtr<FHttpServerResponse>& Response,
		const FHttpServerRequest& Request,
		const FAPIEntry& SelfEntry,
		UAPIRoute* Parent,
		UAPIInstance* Instance,
		FGuid& UserId
	)
{
	return false;
}


AActor* UAPICallback::GetActorWithTag(FName Tag)
{
	AActor* Actor = nullptr;
	if (UWorld* World = _GetWorld())
	{
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* _Actor = *It;
			if (_Actor && _Actor->ActorHasTag(Tag)) { Actor = _Actor; break; }
		}
	}
	return Actor;
}

AActor* UAPICallback::GetActorOfClass(TSubclassOf<AActor> ActorClass)
{
	AActor* Actor = nullptr;
	if (UWorld* World = _GetWorld())
	{
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* _Actor = *It;
			if (_Actor && _Actor->IsA(ActorClass)) { Actor = _Actor; break; }
		}
	}
	return Actor;
}

AActor* UAPICallback::GetActorOfClassWithTag(TSubclassOf<AActor> ActorClass, FName Tag)
{
	AActor* Actor = nullptr;
	if (UWorld* World = _GetWorld())
	{
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* _Actor = *It;
			if (_Actor && _Actor->ActorHasTag(Tag) && _Actor->IsA(ActorClass)) { Actor = _Actor; break; }
		}
	}
	return Actor;
}

void UAPICallback::GetAllActorsWithTag(FName Tag, TArray<AActor*>& OutActors)
{
	if (UWorld* World = _GetWorld())
	{
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (Actor && Actor->ActorHasTag(Tag)) { OutActors.Add(Actor); }
		}
	}
}

void UAPICallback::GetAllActorsOfClass(TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors)
{
	if (UWorld* World = _GetWorld())
	{
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (Actor && Actor->IsA(ActorClass)) { OutActors.Add(Actor); }
		}
	}
}

void UAPICallback::GetAllActorsOfClassWithTag(TSubclassOf<AActor> ActorClass, FName Tag, TArray<AActor*>& OutActors)
{
	if (UWorld* World = _GetWorld())
	{
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (Actor && Actor->ActorHasTag(Tag) && Actor->IsA(ActorClass)) { OutActors.Add(Actor); }
		}
	}
}