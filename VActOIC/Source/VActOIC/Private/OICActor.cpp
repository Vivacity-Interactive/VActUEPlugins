#include "OICActor.h"

AOICActor::AOICActor()
{
	//bIsEditorOnlyActor = true;
	OICComponent = CreateDefaultSubobject<UOICComponent>(
		MakeUniqueObjectName(this, UOICComponent::StaticClass(), TEXT("OIC")));

	DefaultSceneRoot = CreateDefaultSubobject<USceneComponent>(
		MakeUniqueObjectName(this, USceneComponent::StaticClass(), TEXT("DefaultSceneRoot")));
	RootComponent = DefaultSceneRoot;
}


void AOICActor::BeginPlay()
{
	Super::BeginPlay();
	
}

