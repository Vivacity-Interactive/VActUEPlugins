#include "APIComponent.h"

UAPIComponent::UAPIComponent()
	: bUseTickInterval(false)
	, APITickDuration(1.0f)
{
	PrimaryComponentTick.bCanEverTick = true;

}


void UAPIComponent::BeginPlay()
{
	Super::BeginPlay();
	if (APIInstance)
	{
		APIInstance->Init();
	}
}

void UAPIComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	if (APIInstance)
	{
		APIInstance->DeInit();
	}
}


void UAPIComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	
	APITickTime += DeltaTime;
	const bool bDue = APITickTime >= APITickDuration;
	
	const bool bTick = APIInstance != nullptr && (!bUseTickInterval || bDue);
	if (bTick)
	{
		APIInstance->Tick(DeltaTime);
	}

	if (bDue) { APITickTime = 0.0f; }
}

