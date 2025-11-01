#include "STTActor.h"

ASTTActor::ASTTActor()
{
	PrimaryActorTick.bCanEverTick = true;

	STTComponent = CreateDefaultSubobject<USTTComponent>(
		MakeUniqueObjectName(this, USTTComponent::StaticClass(), TEXT("STT")));

	DefaultSceneRoot = CreateDefaultSubobject<USceneComponent>(
		MakeUniqueObjectName(this, USceneComponent::StaticClass(), TEXT("DefaultSceneRoot")));
	
	RootComponent = DefaultSceneRoot;

}

void ASTTActor::BeginPlay()
{
	Super::BeginPlay();
	
}

