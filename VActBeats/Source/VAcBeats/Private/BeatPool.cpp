#include "BeatPool.h"

void UBeatPool::Init()
{
	for (const FBeatEntity& Entity : Entities)
	{
		const bool bAdd = Entity.Name != NAME_None && Entity.Id >= 0;
		if (bAdd)
		{
			NameToId.Add(Entity.Name, Entity.Id);
			IdToClaim.Add(Entity.Id, FBeatClaim(Entity));
		}
	}
}
