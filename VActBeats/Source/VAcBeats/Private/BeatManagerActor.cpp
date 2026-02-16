#include "BeatManagerActor.h"

ABeatManagerActor::ABeatManagerActor()
{
	PrimaryActorTick.bCanEverTick = true;
	ActiveScenarioIndex = 0;

}

void ABeatManagerActor::BeginPlay()
{
	Super::BeginPlay();
	
	Scenarios.SetNum(ScenarioClasses.Num());
	int32 Index = 0;
	for (const TSubclassOf<UBeatPool>& ScenarioClass : ScenarioClasses)
	{
		UBeatPool* Scenario = NewObject<UBeatPool>(this, ScenarioClass);
		Scenario->InitBeatPool();
		Scenario->AddToRoot();
		Scenarios[Index] = Scenario;
		++Index;
	}
}


void ABeatManagerActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void ABeatManagerActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	for (UBeatPool* Scenario : Scenarios) { if (Scenario) { Scenario->RemoveFromRoot(); Scenario = nullptr; } }
}

int32 ABeatManagerActor::PollBeat(FOptionEntry& Into, int32 Self, int32 Target, const FBeatVector& Point, int32 Index, UActorComponent* Component)
{
	UBeatPool* Scenario = GetActiveScenario();
	return -1;
}


int32 ABeatManagerActor::PollBeats(TArray<FOptionEntry>& Into, int32 Self, int32 Target, const FBeatVector& Point, int32 MaxCount, int32 Offset, UActorComponent* Component)
{
	UBeatPool* Scenario = GetActiveScenario();
	return -1;
}

int32 ABeatManagerActor::SelectBeat(FBeatVector& IntoPoint, const FOptionEntry& Choise, bool bSetUsed, bool bSetSkip)
{
	UBeatPool* Scenario = GetActiveScenario();
	return -1;
}

int32 ABeatManagerActor::SelectBeat(FBeatVector& IntoPoint, int32 Index, bool bSetUsed, bool bSetSkip)
{
	UBeatPool* Scenario = GetActiveScenario();
	return -1;
}

int32 ABeatManagerActor::ResolveId(FName ByName)
{
	int32 Id = -1;
	UBeatPool* Scenario = GetActiveScenario();
	if (Scenario) { Id = Scenario->ResolveId(ByName); }
	return Id;
}