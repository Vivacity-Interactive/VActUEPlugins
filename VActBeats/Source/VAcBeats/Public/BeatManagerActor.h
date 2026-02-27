#pragma once

#include "VActBeatsTypes.h"
#include "BeatsProfile.h"
#include "Components/ActorComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BeatManagerActor.generated.h"

//DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnBeatChanged, const FBeatTime&, GlobalBeatState);

//template<typename UserClass>
//using TOnBeatChangedSignature = FOnBeatChanged::FDelegate::template TMethodPtrResolver< UserClass >::FMethodPtr;//void (UserClass::*)(const FGlobalBeatState&);//

class UBeatsProfile;

UCLASS()
class VACTBEATS_API ABeatManagerActor : public AActor
{
	GENERATED_BODY()

public:
	//UPROPERTY(BlueprintAssignable, Category = "Events")
	//FOnBeatChanged OnBeatChanged;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Beat Pool | Classes")
	TArray<TSubclassOf<UBeatsProfile>> ScenarioClasses;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Beat Pool")
	int32 ActiveScenarioIndex;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Beat Pool")
	TArray<TObjectPtr<UBeatsProfile>> Scenarios;

public:
	ABeatManagerActor();

	/*int32 PollBeat(FOptionEntry& Into, int32 Self, int32 Target, const FBeatVector& Point, int32 Index = -1, UActorComponent* Component = nullptr);

	int32 PollBeats(TArray<FOptionEntry>& Into, int32 Self, int32 Target, const FBeatVector& Point, int32 MaxCount = -1, int32 Offset = 0, UActorComponent* Component = nullptr);

	int32 SelectBeat(FBeatVector& IntoPoint, int32 Index, bool bSetUsed = true, bool bSetSkip = false);

	int32 SelectBeat(FBeatVector& IntoPoint, const FOptionEntry& Choise, bool bSetUsed = true, bool bSetSkip = false);

	int32 ResolveId(FName ByName);*/

	template<typename UserClass, typename FuncType>
	FBeatClaim* ClaimEntity(int32 ById, UserClass* Claimer, typename FuncType Method, bool bNearestUnclaimedMatch = false)
	{
		FBeatClaim* Claim = nullptr;
		UBeatsProfile* Scenario = GetActiveScenario();
#if WITH_EDITORONLY_DATA
		UE_LOG(LogTemp, Log, TEXT("'%s' ClaimEntity Scenario=%d."), *GetNameSafe(this), (Scenario ? 1 : -1));
#endif
		if (Scenario)
		{
			Claim = Scenario->ClaimEntity(ById, bNearestUnclaimedMatch);
#if WITH_EDITORONLY_DATA
			UE_LOG(LogTemp, Log, TEXT("\t\t'%s' ClaimEntity Claim=%d \"%s\"."), *GetNameSafe(this), (Claim ? 1 : -1), (Claim ? *Claim->Entity.Title : TEXT("")));
#endif
			//if (Claim) { OnBeatChanged.AddDynamic(Claimer, Method); }
		}

		return Claim;
	}

	FORCEINLINE bool ValidInteraction(const FBeatClaim& Self, const FBeatClaim& Target)
	{
		return ValidInteraction(Self.Entity->Id, Target.Entity->Id);
	}

	FORCEINLINE bool ValidInteraction(int32 Self, int32 Target)
	{
		UBeatsProfile* Scenario = GetActiveScenario();
		return Scenario && Scenario->HasInteraction(Self, Target);
	}

	FORCEINLINE UBeatsProfile* GetActiveScenario()
	{
		return Scenarios.IsValidIndex(ActiveScenarioIndex) ? Scenarios[ActiveScenarioIndex] : nullptr;
	}

protected:
	virtual void BeginPlay() override;

public:
	virtual void Tick(float DeltaTime) override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

};
