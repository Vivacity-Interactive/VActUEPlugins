#include "VActBeatsTypes.h"

FBeatEntity::FBeatEntity()
	: Id(-1)
	, Name(NAME_None)
	, Title(TEXT(""))
{

}

FBeatClaim::FBeatClaim()
	: Entity()
	, Actor(nullptr)
	//, Componnet(nullptr)
{

}

FBeatClaim::FBeatClaim(const FBeatEntity& InEntity, AActor* InActor = nullptr)
	: Entity(Entity)
	, Actor(InActor)
	//, Componnet(nullptr)
{

}

FBeatMeta::FBeatMeta()
	: Name(NAME_None)
	, Self(-1)
	, Context(-1)
	, Class(-1)
{

}