#include "BeatsProfile.h"

const TArray<FName> UBeatsProfile::ModeTypes = {
	TEXT("None"),
	TEXT("Scalar"),
	TEXT("Vector")
};

const TMap<FName, EBeatMode> UBeatsProfile::MapModeTypes = {
	{ ModeTypes[(int32)EBeatMode::None], EBeatMode::None },
	{ ModeTypes[(int32)EBeatMode::Scalar], EBeatMode::Scalar },
	{ ModeTypes[(int32)EBeatMode::Vector], EBeatMode::Vector }
};

void UBeatsProfile::InitClaims(bool bClear)
{
	if (bClear)
	{
		NameToId.Reset();
		IdToClaim.Reset();
	}
	
	for (const FBeatEntity& Entity : Entities)
	{
		const bool bAdd = Entity.Name != NAME_None && Entity.Id >= 0;
		if (bAdd)
		{
			NameToId.Add(Entity.Name, Entity.Id);
			IdToClaim.Add(Entity.Id, FBeatClaim(&Entity));
		}
	}
}