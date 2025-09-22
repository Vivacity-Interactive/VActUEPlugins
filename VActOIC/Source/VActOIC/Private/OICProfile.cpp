#include "OICProfile.h"

const TArray<FName> FOICObject::AssetTypes = {
	TEXT("None"),
	TEXT("Actor"),
	TEXT("Mesh"),
	TEXT("Particle"),
	TEXT("Data"),
	TEXT("Audio"),
	TEXT("System"),
	TEXT("Collider"),
};

const TMap<FName, EOICAsset> FOICObject::MapAssetType = {
	{ AssetTypes[(int32)EOICAsset::None], EOICAsset::None },
	{ AssetTypes[(int32)EOICAsset::Actor], EOICAsset::Actor },
	{ AssetTypes[(int32)EOICAsset::Mesh], EOICAsset::Mesh },
	{ AssetTypes[(int32)EOICAsset::Particle], EOICAsset::Particle },
	{ AssetTypes[(int32)EOICAsset::Data], EOICAsset::Data },
	{ AssetTypes[(int32)EOICAsset::Audio], EOICAsset::Audio },
	{ AssetTypes[(int32)EOICAsset::System], EOICAsset::System },
	{ AssetTypes[(int32)EOICAsset::Collider], EOICAsset::Collider },
};

const TArray<FName> FOICObject::ShapeTypes = {
	TEXT("None"),
	TEXT("Box"),
	TEXT("Sphere"),
	TEXT("Capsule"),
	TEXT("Convex"),
};

const TMap<FName, EOICShape> FOICObject::MapShapeType = {
	{ ShapeTypes[(int32)EOICShape::None], EOICShape::None },
	{ ShapeTypes[(int32)EOICShape::Box], EOICShape::Box },
	{ ShapeTypes[(int32)EOICShape::Sphere], EOICShape::Sphere },
	{ ShapeTypes[(int32)EOICShape::Capsule], EOICShape::Capsule },
	{ ShapeTypes[(int32)EOICShape::Convex], EOICShape::Convex },
};

const TArray<FName> FOICValue::ValueTypes = {
	TEXT("None"),
	TEXT("Name"),
	TEXT("String"),
	TEXT("Bool"),
	TEXT("Float"),
	TEXT("Float2"),
	TEXT("Float3"),
	TEXT("Float4"),
	TEXT("Float5"),
	TEXT("Float6"),
	TEXT("Floats"),
	TEXT("Int"),
	TEXT("Int2"),
	TEXT("Int3"),
	TEXT("Int4"),
	TEXT("Int5"),
	TEXT("Int6"),
	TEXT("Ints"),
	TEXT("Names"),
	TEXT("Strings"),
	TEXT("Asset"),
};

const TMap<FName, EOICValue> FOICValue::MapValueType = {
	{ ValueTypes[(int32)EOICValue::None], EOICValue::None},
	{ ValueTypes[(int32)EOICValue::Name], EOICValue::Name},
	{ ValueTypes[(int32)EOICValue::String], EOICValue::String},
	{ ValueTypes[(int32)EOICValue::Bool], EOICValue::Bool},
	{ ValueTypes[(int32)EOICValue::Float], EOICValue::Float},
	{ ValueTypes[(int32)EOICValue::Float2], EOICValue::Float2},
	{ ValueTypes[(int32)EOICValue::Float3], EOICValue::Float3},
	{ ValueTypes[(int32)EOICValue::Float4], EOICValue::Float4},
	{ ValueTypes[(int32)EOICValue::Float5], EOICValue::Float5},
	{ ValueTypes[(int32)EOICValue::Float6], EOICValue::Float6},
	{ ValueTypes[(int32)EOICValue::Floats], EOICValue::Ints},
	{ ValueTypes[(int32)EOICValue::Int], EOICValue::Int},
	{ ValueTypes[(int32)EOICValue::Int2], EOICValue::Int2},
	{ ValueTypes[(int32)EOICValue::Int3], EOICValue::Int3},
	{ ValueTypes[(int32)EOICValue::Int4], EOICValue::Int4},
	{ ValueTypes[(int32)EOICValue::Int5], EOICValue::Int5},
	{ ValueTypes[(int32)EOICValue::Int6], EOICValue::Int6},
	{ ValueTypes[(int32)EOICValue::Ints], EOICValue::Ints},
	{ ValueTypes[(int32)EOICValue::Names], EOICValue::Names},
	{ ValueTypes[(int32)EOICValue::Strings], EOICValue::Strings},
	{ ValueTypes[(int32)EOICValue::Asset], EOICValue::Asset},
};

FOICProfileEntry::FOICProfileEntry()
{
	bUpdateMetas = bInitialized = bSkip = false;
	bTracked = true;
	bUseStaticMeshInstances = true;
	bClear = false;
}