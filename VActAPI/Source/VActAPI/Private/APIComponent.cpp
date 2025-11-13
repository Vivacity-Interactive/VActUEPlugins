#include "APIComponent.h"

UAPIComponent::UAPIComponent()
{
	PrimaryComponentTick.bCanEverTick = true;

}


void UAPIComponent::BeginPlay()
{
	Super::BeginPlay();
	
}


void UAPIComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

