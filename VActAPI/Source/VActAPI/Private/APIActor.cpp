#include "APIActor.h"

AAPIActor::AAPIActor()
{
	PrimaryActorTick.bCanEverTick = true;

	APIComponent = CreateDefaultSubobject<UAPIComponent>(
		MakeUniqueObjectName(this, UAPIComponent::StaticClass(), TEXT("API")));

	DefaultSceneRoot = CreateDefaultSubobject<USceneComponent>(
		MakeUniqueObjectName(this, USceneComponent::StaticClass(), TEXT("DefaultSceneRoot")));

	RootComponent = DefaultSceneRoot;

}

void AAPIActor::BeginPlay()
{
	Super::BeginPlay();
	
}

