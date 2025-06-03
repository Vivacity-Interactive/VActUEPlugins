#include "OICProfileCustomization.h"

#include "DetailCategoryBuilder.h"
#include "DetailLayoutBuilder.h"
#include "DetailWidgetRow.h"

#include "OICProfile.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SFilePathPicker.h"
#include "Widgets/Input/SDirectoryPicker.h"
#include "Widgets/Input/SComboBox.h"

#include "VActFileUtils.h"
#include "VActFileTypes.h"
#include "VActFiles.h"

#include "Misc/Optional.h"

#define LOCTEXT_NAMESPACE "OICProfileCustomization"

const FString FOICProfileCustomization::TypeFilter = TEXT("All OIC (*.*oic)|*.*oic|OICProfile files (*.oic)|*.oic|Binary OICProfile files (*.boic)|*.boic|Compact OICProfile files (*.coic)|*.coic|All files (*.*)|*.*");

const FName FOICProfileCustomization::EditCategory = TEXT("OICProfileIO");

const TArray<FName> FOICProfileCustomization::PropertyNames = {
	FName(TEXT("Notes")),
	FName(TEXT("Title")),
	FName(TEXT("Name")),
	FName(TEXT("Instances")),
	FName(TEXT("Objects")),
	FName(TEXT("Transform")),
	FName(TEXT("Metas")),
	FName(TEXT("Meta")),
	FName(TEXT("Type")),
	FName(TEXT("Value")),
	FName(TEXT("Axis")),
	FName(TEXT("Version")),
	FName(TEXT("Properties")),
	FName(TEXT("Asset")),
};

enum class _EPropName
{
	Notes = 0,
	Title,
	Name,
	Instances,
	Objects,
	Transform,
	Metas,
	Meta,
	Type,
	Value,
	Axis,
	Version,
	Properties,
	Asset,
};

TSharedRef<IDetailCustomization> FOICProfileCustomization::MakeInstance()
{
    return MakeShareable(new FOICProfileCustomization);
}

bool FOICProfileCustomization::ParseFromCursorsBinary(class UOICProfile* InContext, const FVActParseRoot& Root)
{
#if WITH_EDITORONLY_DATA
	int32 Index = 0; bool _b_Index = false;
	for (const FVActParseCursor& Cursor : Root.Cursors)
	{
		_b_Index = _DEBUG_VActParseInfo::TokenNames.IsValidIndex((int32)Cursor.Token);
		UE_LOG(LogTemp, Log, TEXT("[%2d] %7s(%d): %.*s"), Index++, (_b_Index ? *_DEBUG_VActParseInfo::TokenNames[(int32)Cursor.Token] : TEXT("")), (int32)Cursor.Token, Cursor.Len(), Cursor.From);
	}
#endif
	bool bValid = false, bEnd = false, bNext = true, _bNext = true;
	FVActParseRootIt It = FVActParseRootIt(Root);

#if WITH_EDITORONLY_DATA
	_b_Index = _DEBUG_VActParseInfo::TokenNames.IsValidIndex((int32)It.Token());
	UE_LOG(LogTemp, Log, TEXT("%d \t[%2d] %7s(%d)"), (bValid ? 1 : 0), It.Index, (_b_Index ? *_DEBUG_VActParseInfo::TokenNames[(int32)It.Token()] : TEXT("")), (int32)It.Token());
#endif

	return bValid;
}

bool FOICProfileCustomization::ParseFromCursorsCompact(class UOICProfile* InContext, const FVActParseRoot& Root)
{
#if WITH_EDITORONLY_DATA
	int32 Index = 0; bool _b_Index = false;
	for (const FVActParseCursor& Cursor : Root.Cursors)
	{
		_b_Index = _DEBUG_VActParseInfo::TokenNames.IsValidIndex((int32)Cursor.Token);
		UE_LOG(LogTemp, Log, TEXT("[%2d] %7s(%d): %.*s"), Index++, (_b_Index ? *_DEBUG_VActParseInfo::TokenNames[(int32)Cursor.Token] : TEXT("")), (int32)Cursor.Token, Cursor.Len(), Cursor.From);
	}
#endif
	bool bValid = false, bEnd = false, bNext = true, _bNext = true;
	FVActParseRootIt It = FVActParseRootIt(Root);

#if WITH_EDITORONLY_DATA
	_b_Index = _DEBUG_VActParseInfo::TokenNames.IsValidIndex((int32)It.Token());
	UE_LOG(LogTemp, Log, TEXT("%d \t[%2d] %7s(%d)"), (bValid ? 1 : 0), It.Index, (_b_Index ? *_DEBUG_VActParseInfo::TokenNames[(int32)It.Token()] : TEXT("")), (int32)It.Token());
#endif

	return bValid;
}

bool FOICProfileCustomization::ParseFromCursorsJsonStrict(class UOICProfile* InContext, const FVActParseRoot& Root)
{
	bool bValid = true, bEnd = false, bNext = true;
	FVActParseRootIt It = FVActParseRootIt(Root);

	InContext->Objects.Empty();
	InContext->Instances.Empty();
	InContext->Metas.Empty();

	bValid = It.Token() == EVActParseToken::_Struct;
	bValid &= FVactTextParseUtils::Parse(InContext->Type, It += 2);
	bValid &= FVactTextParseUtils::Parse(InContext->Version, It += 2); // needs to move before notes and title
	bValid &= FVactTextParseUtils::Parse(InContext->Axis, It += 2); // needs to move before notes and title
#if WITH_EDITORONLY_DATA
	bValid &= FVactTextParseUtils::Parse(InContext->Notes, It += 2);
	bValid &= FVactTextParseUtils::Parse(InContext->Title, It += 2);
#else
	It += 6; // skipp notes and title
#endif
	bValid &= FVactTextParseUtils::Parse(InContext->Name, It += 2);

	// Property Array Objects
	bValid &= (It += 2) && It.Token() == EVActParseToken::_Array && ++It;
	bNext = It.Token() != EVActParseToken::Array;
	while (bValid && bNext && It)
	{
		FOICObject Object;
		bValid &= FVactTextParseUtils::ParseEnum(Object.Type, It += 2, FOICObject::MapAssetType);
		switch (Object.Type)
		{
		case EOICAsset::Actor:
			bValid &= FVactTextParseUtils::ParseClass(Object.Actor, It += 2, AActor::StaticClass(), InContext);
			break;
		case EOICAsset::Particle:
		case EOICAsset::Mesh:
			bValid &= FVactTextParseUtils::ParseObject(Object.Mesh, It += 2, UStaticMesh::StaticClass(), InContext);
			break;
		case EOICAsset::System:
		case EOICAsset::Data:
			bValid &= FVactTextParseUtils::ParseClass(Object.Data, It += 2, UStaticMesh::StaticClass(), InContext);
			break;
		case EOICAsset::Audio:
			bValid &= FVactTextParseUtils::ParseObject(Object.Audio, It += 2, UStaticMesh::StaticClass(), InContext);
			break;
		default: bValid = false; break; // TODO find way to support skipping
		}
		bValid &= FVactTextParseUtils::Parse(Object.Meta, It += 2);
		
		if (bValid) { InContext->Objects.Add(Object); }
		bNext = (It += 2) && It.Token() != EVActParseToken::Array;
	}
	bValid &= It.Token() == EVActParseToken::Array && ++It;

	// Property Array Instances
	bValid &= (It += 2) && It.Token() == EVActParseToken::_Array && ++It;
	bNext = It.Token() != EVActParseToken::Array;
	while (bValid && bNext && It)
	{
		FOICInstance Instance;
		bValid &= It.Token() == EVActParseToken::_Tuple && ++It
			&& FVactTextParseUtils::Parse(Instance.Id, It)
			&& FVactTextParseUtils::Parse(Instance.Object, It)
			&& FVactTextParseUtils::Parse(Instance.Parent, It)
			&& FVactTextParseUtils::Parse(Instance.Meta, It)
			&& FVactTextParseUtils::Parse(Instance.Transform, It)
			&& It.Token() == EVActParseToken::Tuple && ++It;

		if (bValid) { InContext->Instances.Add(Instance); }
		bNext = It.Token() != EVActParseToken::Array;
	}
	bValid &= It.Token() == EVActParseToken::Array && ++It;

	// Property Array Metas
	bValid &= (It += 2) && It.Token() == EVActParseToken::_Array && ++It;
	bNext = It.Token() != EVActParseToken::Array;
	while (bValid && bNext && It)
	{
		FOICMeta Meta;
		bValid &= _ParseOICMeta(Meta, It, InContext);
		if (bValid) { InContext->Metas.Add(Meta); }
		bNext = It.Token() != EVActParseToken::Array;
	}
	bValid &= It.Token() == EVActParseToken::Array && ++It;
	
	//bValid &= It.Token() == EVActParseToken::Struct;

	InContext->InstancesCount = InContext->Instances.Num();
	return bValid;
}

bool FOICProfileCustomization::_ParseOICMeta(FOICMeta& Into, FVActParseRootIt& It, UObject* InOuter)
{
	bool bValid = It && It.Token() == EVActParseToken::_Array && ++It, _bNext = true;
	bool bNext = It.Token() != EVActParseToken::Array;
	while (bValid && bNext && It)
	{
		FOICMetaEntry Entry;
		bValid &= It.Token() == EVActParseToken::_Struct;
		bValid &= FVactTextParseUtils::ParseClass(Entry.Asset, (It += 2), UActorComponent::StaticClass(), InOuter);
		bValid &= (It += 2) && It.Token() == EVActParseToken::_Struct && ++It;
		_bNext = It.Token() != EVActParseToken::Struct;
		while (bValid && _bNext && It)
		{
			FOICValue Value;
			FName Key;
			bValid &= FVactTextParseUtils::Parse(Key, It);
			bValid &= It.Token() == EVActParseToken::_Struct;
			bValid &= FVactTextParseUtils::ParseEnum(Value.Type, (It += 2), FOICValue::MapValueType);
			bValid &= (It += 2) || false;
			switch (Value.Type)
			{
			case EOICValue::Name: bValid &= FVactTextParseUtils::ParsePlace(Value.GetValue<FName>(), It, Value.Flag); break;
			case EOICValue::String: bValid &= FVactTextParseUtils::ParsePlace(Value.GetValue<FString>(), It, Value.Flag); break;
			case EOICValue::Bool: bValid &= FVactTextParseUtils::Parse(Value.GetValue<bool>(), It); break;
			case EOICValue::Float: bValid &= FVactTextParseUtils::Parse(Value.GetValue<float>(), It); break;
			case EOICValue::Float2: bValid &= FVactTextParseUtils::ParseTuple(Value._FData, It, 2); break;
			case EOICValue::Float3: bValid &= FVactTextParseUtils::ParseTuple(Value._FData, It, 3); break;
			case EOICValue::Float4: bValid &= FVactTextParseUtils::ParseTuple(Value._FData, It, 4); break;
			case EOICValue::Float5: bValid &= FVactTextParseUtils::ParseTuple(Value._FData, It, 5); break;
			case EOICValue::Float6: bValid &= FVactTextParseUtils::ParseTuple(Value._FData, It, 6); break;
			case EOICValue::Floats: bValid &= FVactTextParseUtils::ParseTuplePlace(Value.GetValue<TArray<float>>(), It, Value.Flag); break;
			case EOICValue::Int: bValid &= FVactTextParseUtils::Parse(Value.GetValue<int32>(), It); break;
			case EOICValue::Int2: bValid &= FVactTextParseUtils::ParseTuple(Value._IData, It, 2); break;
			case EOICValue::Int3: bValid &= FVactTextParseUtils::ParseTuple(Value._IData, It, 3); break;
			case EOICValue::Int4: bValid &= FVactTextParseUtils::ParseTuple(Value._IData, It, 4); break;
			case EOICValue::Int5: bValid &= FVactTextParseUtils::ParseTuple(Value._IData, It, 5); break;
			case EOICValue::Int6: bValid &= FVactTextParseUtils::ParseTuple(Value._IData, It, 6); break;
			case EOICValue::Ints: bValid &= FVactTextParseUtils::ParseTuplePlace(Value.GetValue<TArray<int32>>(), It, Value.Flag); break;
			case EOICValue::Names: bValid &= FVactTextParseUtils::ParseTuplePlace(Value.GetValue<TArray<FName>>(), It, Value.Flag); break;
			case EOICValue::Strings: bValid &= FVactTextParseUtils::ParseTuplePlace(Value.GetValue<TArray<FString>>(), It, Value.Flag); break;
			case EOICValue::Asset: 
				bValid &= FVactTextParseUtils::ParseEnum(Value._Type, It, FOICObject::MapAssetType);
				bValid &= FVactTextParseUtils::ParseTuplePlace(Value.GetValue<TArray<FString>>(), It, Value.Flag);
				break;
			default: bValid = false; break;
			}
			bValid &= ++It && It.Token() == EVActParseToken::Struct && (It += 2);
			if (bValid) { Entry.Properties.Add(Key, Value); }
			_bNext = It.Token() != EVActParseToken::Struct;
		}
		bValid &= It.Token() == EVActParseToken::Struct && ++It;
		if (bValid) { Into.Entries.Add(Entry); }
		bNext = (It += 2) && It.Token() != EVActParseToken::Array;
	}
	bValid &= It.Token() == EVActParseToken::Array && ++It;
	return bValid;
}

bool FOICProfileCustomization::ParseFromCursorsJson(class UOICProfile* InContext, const FVActParseRoot& Root)
{
	return true;
}

bool FOICProfileCustomization::ComposeToCursorsCompact(class UOICProfile* InContext, FVActComposeRoot& Root)
{
	FVActComposeUtils::ComposeStructOpen(Root.Cursors);

	FVActComposeUtils::Compose(InContext->Type, Root.Cursors);
	FVActComposeUtils::Compose(InContext->Version, Root.Cursors);
	FVActComposeUtils::Compose(InContext->Axis, Root.Cursors);
#if WITH_EDITORONLY_DATA
	FVActComposeUtils::Compose(InContext->Notes, Root.Cursors);
	FVActComposeUtils::Compose(InContext->Title, Root.Cursors);
#endif
	FVActComposeUtils::Compose(InContext->Name, Root.Cursors);

	// Property Array Objects
	FVActComposeUtils::ComposeArrayOpen(Root.Cursors);
	for (const FOICObject& Object : InContext->Objects)
	{
		FVActComposeUtils::ComposeStructOpen(Root.Cursors);
		FVActComposeUtils::ComposeEnum(Object.Type, Root.Cursors, FOICObject::AssetTypes);
		switch (Object.Type)
		{
		case EOICAsset::Actor: FVActComposeUtils::Compose(Object.Actor->GetPathName(), Root.Cursors); break;
		case EOICAsset::Particle:
		case EOICAsset::Mesh: FVActComposeUtils::Compose(Object.Mesh->GetPathName(), Root.Cursors); break;
		case EOICAsset::System:
		case EOICAsset::Data: FVActComposeUtils::Compose(Object.Data->GetPathName(), Root.Cursors); break;
		case EOICAsset::Audio: FVActComposeUtils::Compose(Object.Audio->GetPathName(), Root.Cursors); break;
		default: /*FVActComposeUtils::Compose(TEXT(""), Root.Cursors);*/ break;
		}
		FVActComposeUtils::Compose(Object.Meta, Root.Cursors);
		FVActComposeUtils::ComposeStruct(Root.Cursors);
	}
	FVActComposeUtils::ComposeArray(Root.Cursors);

	// Property Array Instances
	FVActComposeUtils::ComposeArrayOpen(Root.Cursors);
	for (const FOICInstance& Instance : InContext->Instances)
	{
		FVActComposeUtils::ComposeTupleOpen(Root.Cursors);
		FVActComposeUtils::Compose(Instance.Id, Root.Cursors);
		FVActComposeUtils::Compose(Instance.Object, Root.Cursors);
		FVActComposeUtils::Compose(Instance.Parent, Root.Cursors);
		FVActComposeUtils::Compose(Instance.Meta, Root.Cursors);
		FVActComposeUtils::Compose(Instance.Transform, Root.Cursors);
		FVActComposeUtils::ComposeTuple(Root.Cursors);
	}
	FVActComposeUtils::ComposeArray(Root.Cursors);
	
	// Property Array Metas
	FVActComposeUtils::ComposeArrayOpen(Root.Cursors);
	for (const FOICMeta& Meta : InContext->Metas)
	{
		FVActComposeUtils::ComposeArrayOpen(Root.Cursors);
		for (const FOICMetaEntry& Entry : Meta.Entries)
		{
			FVActComposeUtils::ComposeStructOpen(Root.Cursors);
			FVActComposeUtils::Compose(Entry.Asset->GetPathName(), Root.Cursors);
			FVActComposeUtils::ComposeStructOpen(Root.Cursors);
			for (const TPair<FName, FOICValue>& Pair : Entry.Properties)
			{
				FVActComposeUtils::Compose(Pair.Key, Root.Cursors);
				FVActComposeUtils::ComposeStructOpen(Root.Cursors);
				FVActComposeUtils::ComposeEnum(Pair.Value.Type, Root.Cursors, FOICValue::ValueTypes);
				switch (Pair.Value.Type)
				{
				case EOICValue::Name: FVActComposeUtils::Compose(Pair.Value.GetValue<FName>(), Root.Cursors); break;
				case EOICValue::String: FVActComposeUtils::Compose(Pair.Value.GetValue<FString>(), Root.Cursors); break;
				case EOICValue::Bool: FVActComposeUtils::Compose(Pair.Value.GetValue<bool>(), Root.Cursors); break;
				case EOICValue::Float: FVActComposeUtils::Compose(Pair.Value.GetValue<float>(), Root.Cursors); break;
				case EOICValue::Float2: FVActComposeUtils::ComposeTuple(Pair.Value._FData, Root.Cursors, 2); break;
				case EOICValue::Float3: FVActComposeUtils::ComposeTuple(Pair.Value._FData, Root.Cursors, 3); break;
				case EOICValue::Float4: FVActComposeUtils::ComposeTuple(Pair.Value._FData, Root.Cursors, 4); break;
				case EOICValue::Float5: FVActComposeUtils::ComposeTuple(Pair.Value._FData, Root.Cursors, 5); break;
				case EOICValue::Float6: FVActComposeUtils::ComposeTuple(Pair.Value._FData, Root.Cursors, 6); break;
				case EOICValue::Floats: FVActComposeUtils::ComposeTuple(Pair.Value.GetValue<TArray<float>>(), Root.Cursors); break;
				case EOICValue::Int: FVActComposeUtils::Compose(Pair.Value.GetValue<int32>(), Root.Cursors); break;
				case EOICValue::Int2: FVActComposeUtils::ComposeTuple(Pair.Value._IData, Root.Cursors, 2); break;
				case EOICValue::Int3: FVActComposeUtils::ComposeTuple(Pair.Value._IData, Root.Cursors, 3); break;
				case EOICValue::Int4: FVActComposeUtils::ComposeTuple(Pair.Value._IData, Root.Cursors, 4); break;
				case EOICValue::Int5: FVActComposeUtils::ComposeTuple(Pair.Value._IData, Root.Cursors, 5); break;
				case EOICValue::Int6: FVActComposeUtils::ComposeTuple(Pair.Value._IData, Root.Cursors, 6); break;
				case EOICValue::Ints: FVActComposeUtils::ComposeTuple(Pair.Value.GetValue<TArray<int32>>(), Root.Cursors); break;
				case EOICValue::Names: FVActComposeUtils::ComposeTuple(Pair.Value.GetValue<TArray<FName>>(), Root.Cursors); break;
				case EOICValue::Strings: FVActComposeUtils::ComposeTuple(Pair.Value.GetValue<TArray<FString>>(), Root.Cursors); break;
				case EOICValue::Asset: 
					FVActComposeUtils::ComposeEnum(Pair.Value._Type, Root.Cursors, FOICObject::AssetTypes);
					FVActComposeUtils::Compose(Pair.Value.GetValue<FString>(), Root.Cursors);
					break;
				default:/*FVActComposeUtils::ComposeNull(Root.Cursors);*/ break;
				}
				FVActComposeUtils::ComposeStruct(Root.Cursors);
				//FVActComposeUtils::ComposeProperty(Root.Cursors);
			}
			FVActComposeUtils::ComposeStruct(Root.Cursors);
			FVActComposeUtils::ComposeStruct(Root.Cursors);
		}
		FVActComposeUtils::ComposeArray(Root.Cursors);
	}
	FVActComposeUtils::ComposeArray(Root.Cursors);

	FVActComposeUtils::ComposeStruct(Root.Cursors);

	return true;
}

bool FOICProfileCustomization::ComposeToCursors(class UOICProfile* InContext, FVActComposeRoot& Root)
{
	return true;
}

#undef LOCTEXT_NAMESPACE