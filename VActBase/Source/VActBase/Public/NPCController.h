// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "QueueHandler.h"
#include "ListHandler.h"

#include "VActCharacter.h"
#include "ActionEvent.h"

#include "CoreMinimal.h"
#include "AIController.h"
#include "NPCController.generated.h"

class UActionEvent;

UCLASS()
class VACTBASE_API ANPCController : public AAIController
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	float NextReactionTime;

	UPROPERTY(EditAnywhere)
	TArray<TSubclassOf<UActionEvent>> ActionEventClasses;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TWeakObjectPtr<AVActCharacter> NPC;

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TArray<TObjectPtr<UActionEvent>> ActiveActionEvents;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TArray<TObjectPtr<UActionEvent>> ActionEventQueue;

	TListHandler<TObjectPtr<UActionEvent>> _ActiveActionEvents;

	TQueueHandler<TObjectPtr<UActionEvent>> _ActionEventQueue;

public:
	ANPCController();

	virtual void Think();

	void React();

	void InitForNewNPC();

protected:
	virtual void Tick(float DeltaTime) override;

	virtual void BeginPlay() override;

	virtual void OnPossess(APawn* InPawn) override;
};
