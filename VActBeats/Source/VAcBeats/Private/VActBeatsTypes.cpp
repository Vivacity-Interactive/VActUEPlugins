#include "VActBeatsTypes.h"

const TArray<FName> FBeatAsset::AssetTypes = {
	TEXT("None"),
	TEXT("Asset"),
	TEXT("Text")
};

const TMap<FName, EBeatAsset> FBeatAsset::MapAssetTypes = {
	{ AssetTypes[(int32)EBeatAsset::None], EBeatAsset::None },
	{ AssetTypes[(int32)EBeatAsset::Asset], EBeatAsset::Asset },
	{ AssetTypes[(int32)EBeatAsset::Text], EBeatAsset::Text }
};

const TArray<FName> FBeatPrototype::PrototypeTypes = {
	TEXT("None"),
	TEXT("Point"),
	TEXT("Vector"),
	TEXT("Interval")
};

const TMap<FName, EBeatPrototype> FBeatPrototype::MapPrototypeTypes = {
	{ PrototypeTypes[(int32)EBeatPrototype::None], EBeatPrototype::None },
	{ PrototypeTypes[(int32)EBeatPrototype::Point], EBeatPrototype::Point },
	{ PrototypeTypes[(int32)EBeatPrototype::Vector], EBeatPrototype::Vector },
	{ PrototypeTypes[(int32)EBeatPrototype::Interval], EBeatPrototype::Interval }
};

FBeatAsset::FBeatAsset()
	: Type(EBeatAsset::None)
	, Name(NAME_None)
	, _Ptr(nullptr)
{
}

FBeatEffect::FBeatEffect()
	: Vector(nullptr)
	, Effector(nullptr)
{
}

FBeatEntity::FBeatEntity()
	: Title(TEXT(""))
	, Id(-1)
	, Name(NAME_None)
{
}

FBeatClaim::FBeatClaim()
	: FBeatClaim(nullptr, nullptr)
{
}

FBeatClaim::FBeatClaim(const FBeatEntity* InEntity, AActor* InActor)
	: Entity(InEntity)
	, Title(InEntity ? InEntity->Title : TEXT(""))
	, Actor(InActor)
	//, Componnet(nullptr)
{
}

FBeatMeta::FBeatMeta()
	: Self(-1)
	, Context(-1)
	, Class(-1)
{
}

FBeatContext::FBeatContext()
	: Title(TEXT(""))
	, Name(NAME_None)
	, Assets()
{
}

FBeatPrototype::FBeatPrototype()
	: Id(-1)
	, Meta()
	, Weight(0.f)
	, Type(EBeatPrototype::None)
	, Coordinate(nullptr)
	, Effect()
	, Contexts()
{
}