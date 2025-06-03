// Fill out your copyright notice in the Description page of Project Settings.
#include "NPCController.h"

#if WITH_EDITORONLY_DATA
#include "Kismet/KismetSystemLibrary.h"
#endif

#include "Misc/AssertionMacros.h"

#include "IterateHandler.h"
#include "Sampler.h"

#include "VActCharacter.h"

class ACharacter;

ANPCController::ANPCController()
{
	_ActiveActionEvents = TListHandler(ActiveActionEvents);
	_ActionEventQueue = TQueueHandler(ActionEventQueue);

	bCanPossessWithoutAuthority = true;
	bStartAILogicOnPossess = true;

	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
}

void ANPCController::BeginPlay()
{
	Super::BeginPlay();
}

void ANPCController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	
	NextReactionTime -= DeltaTime;
	const bool bReact = NextReactionTime <= FLT_EPSILON && NPC != nullptr && NPC.IsValid();
	if (bReact) { React(); NextReactionTime = NPC->ReactionTime; }

	auto It = _ActiveActionEvents.Iterator();
	while (It.Next())
	{
		TObjectPtr<UActionEvent>& ActionEvent = _ActiveActionEvents.Get(It.Index);
		check(ActionEvent);

		const bool bExecute = !ActionEvent->bCompleted && !ActionEvent->bInterupted;
		ActionEvent->Duration += DeltaTime;
		if (bExecute)
		{
			ActionEvent->Execute(DeltaTime);
		}
		else
		{
			_ActiveActionEvents.RemoveNull(It.Index, nullptr);
		}
	}

	// TODO: Figure out how to do this nicely
	//NPC->_Move();
}

void ANPCController::OnPossess(APawn* InPawn)
{
	Super::OnPossess(InPawn);
	
	const bool bValid = InPawn->IsA<AVActCharacter>();
	if (!bValid)
	{
		UnPossess();	
		const bool bNPC = NPC != nullptr && NPC.IsValid();
		if (bNPC) { Possess(NPC.Get()); }
	}

	NPC = Cast<AVActCharacter>(GetPawn());

	InitForNewNPC();
}

void ANPCController::Think()
{
	const bool bQueue = !_ActionEventQueue.IsFull() && !ActionEventClasses.IsEmpty();
	if (bQueue)
	{
		TSampler Sampler(ActionEventClasses);
		TObjectPtr<UActionEvent> ActionEvent = NewObject<UActionEvent>(this, Sampler.Any());
		ActionEvent->Owner = NPC;
		ActionEvent->OwnerController = this;
		_ActionEventQueue.Enqueue(ActionEvent);
	}
}

void ANPCController::React()
{	
#if WITH_EDITORONLY_DATA
	UKismetSystemLibrary::PrintString(this, FString::Printf(TEXT("Reaction %s"), NPC != nullptr ? *NPC->GetName() : *GetName()));
#endif
	const bool bPoll = !_ActiveActionEvents.IsFull() && !_ActionEventQueue.IsEmpty();
	if (bPoll)
	{
		TObjectPtr<UActionEvent> ActionEvent;
		_ActionEventQueue.DequeueNull(ActionEvent, nullptr);
		check(ActionEvent);
		_ActiveActionEvents.Add(ActionEvent);
	}
	
	Think();
}

void ANPCController::InitForNewNPC()
{
	NextReactionTime = NPC->ReactionTime;

	const bool bMultitasking = NPC->MaxMultitasking > 0;
	if (bMultitasking)
	{
		ActiveActionEvents.SetNum(NPC->MaxMultitasking);
	}

	const bool bWorkingMemory = NPC->MaxWorkingMemory > 0;
	if (bWorkingMemory)
	{
		ActionEventQueue.SetNum(NPC->MaxWorkingMemory);
	}

	_ActiveActionEvents = TListHandler(ActiveActionEvents);
	_ActionEventQueue = TQueueHandler(ActionEventQueue);
}