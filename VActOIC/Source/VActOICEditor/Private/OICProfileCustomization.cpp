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
	FName(TEXT("Shape")),
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
	Shape,
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
		//bValid &= FVactTextParseUtils::ParseEnum(Object.Shape, It += 2, FOICObject::MapShapeType);
		switch (Object.Type)
		{
		case EOICAsset::Actor:
			bValid &= FVactTextParseUtils::ParseClass(Object.Actor, It += 2, AActor::StaticClass(), InContext);
			break;
		case EOICAsset::Collider:
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

bool FOICProfileCustomization::EmitToCursorsBinary(class UOICProfile* InContext, FVActEmitRoot& Root)
{
	return true;
}

bool FOICProfileCustomization::EmitToCursorsCompact(class UOICProfile* InContext, FVActEmitRoot& Root)
{
	return true;
}

bool FOICProfileCustomization::EmitToCursorsJson(class UOICProfile* InContext, FVActEmitRoot& Root)
{
	return true;
}

bool FOICProfileCustomization::EmitToCursorsJsonStrict(class UOICProfile* InContext, FVActEmitRoot& Root)
{
	const TCHAR* OICBegin = Root.End();
	FVActTextEmitUtils::EmitStructOpen(Root.Cursors, Root.Source);
	
	FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Type], InContext->Type, Root.Cursors, Root.Source);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
	
	FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Version], InContext->Version, Root.Cursors, Root.Source);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
	
	FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Axis], InContext->Axis, Root.Cursors, Root.Source);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
#if WITH_EDITORONLY_DATA
	FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Notes], InContext->Notes, Root.Cursors, Root.Source);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
	
	FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Title], InContext->Title, Root.Cursors, Root.Source);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
#endif
	FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Name], InContext->Name, Root.Cursors, Root.Source);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);

	// Property Array Objects
	const TCHAR* ObjectsPropBegin = Root.End();
	FVActTextEmitUtils::EmitPropertyOpen(PropertyNames[(int32)_EPropName::Objects], Root.Cursors, Root.Source);
	const TCHAR* ObjectsBegin = Root.End();
	FVActTextEmitUtils::EmitArrayOpen(Root.Cursors, Root.Source);
	const int32 ObjectCount = InContext->Objects.Num();
	for (int32 Index = 0; Index < ObjectCount; ++Index)
	{
		const FOICObject& Object = InContext->Objects[Index];
		const TCHAR* ObjectBegin = Root.End();
		FVActTextEmitUtils::EmitStructOpen(Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::EmitEnumProperty(PropertyNames[(int32)_EPropName::Type], Object.Type, Root.Cursors, Root.Source, FOICObject::AssetTypes);
		FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
		
		//FVActTextEmitUtils::EmitEnumProperty(PropertyNames[(int32)_EPropName::Shape], Object.Shape, Root.Cursors, Root.Source, FOICObject::ShapeTypes);
		//FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
		
		FName _TypeKey = PropertyNames[(int32)_EPropName::Asset];
		switch (Object.Type)
		{
		case EOICAsset::Actor: FVActTextEmitUtils::EmitProperty(_TypeKey, Object.Actor->GetPathName(), Root.Cursors, Root.Source); break;
		case EOICAsset::Collider:
		case EOICAsset::Particle:
		case EOICAsset::Mesh: FVActTextEmitUtils::EmitProperty(_TypeKey, Object.Mesh->GetPathName(), Root.Cursors, Root.Source); break;
		case EOICAsset::System:
		case EOICAsset::Data: FVActTextEmitUtils::EmitProperty(_TypeKey, Object.Data->GetPathName(), Root.Cursors, Root.Source); break;
		case EOICAsset::Audio: FVActTextEmitUtils::EmitProperty(_TypeKey, Object.Audio->GetPathName(), Root.Cursors, Root.Source); break;
		default: /*FVActComposeUtils::Emit(TEXT(""), Root.Cursors, Root.Source);*/ break;
		}
		FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::EmitProperty(PropertyNames[(int32)_EPropName::Meta], Object.Meta, Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::EmitStruct(Root.Cursors, Root.Source, ObjectBegin);
		if (Index < ObjectCount - 1) { FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source); }
	}
	FVActTextEmitUtils::EmitArray(Root.Cursors, Root.Source, ObjectsBegin);
	FVActTextEmitUtils::EmitProperty(Root.Cursors, Root.Source, ObjectsPropBegin);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);

	// Property Array Instances
	const TCHAR* InstancesPropBegin = Root.End();
	FVActTextEmitUtils::EmitPropertyOpen(PropertyNames[(int32)_EPropName::Instances], Root.Cursors, Root.Source);
	const TCHAR* InstancesBegin = Root.End();
	FVActTextEmitUtils::EmitArrayOpen(Root.Cursors, Root.Source);
	const int32 InstanceCount = InContext->Instances.Num();
	for (int32 Index = 0; Index < InstanceCount; ++Index)
	{
		const FOICInstance& Instance = InContext->Instances[Index];
		const TCHAR* InstanceBegin = Root.End();
		FVActTextEmitUtils::EmitTupleOpen(Root.Cursors, Root.Source); 
		
		FVActTextEmitUtils::Emit(Instance.Id, Root.Cursors, Root.Source);
		FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::Emit(Instance.Object, Root.Cursors, Root.Source);
		FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::Emit(Instance.Parent, Root.Cursors, Root.Source);
		FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::Emit(Instance.Meta, Root.Cursors, Root.Source);
		FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);

		FVActTextEmitUtils::Emit(Instance.Transform, Root.Cursors, Root.Source);
		
		FVActTextEmitUtils::EmitTuple(Root.Cursors, Root.Source, InstanceBegin);
		if (Index < InstanceCount - 1) { FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source); }
	}
	FVActTextEmitUtils::EmitArray(Root.Cursors, Root.Source, InstancesBegin);
	FVActTextEmitUtils::EmitProperty(Root.Cursors, Root.Source, InstancesPropBegin);
	FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source);

	// Property Array Metas
	const TCHAR* MetasPropBegin = Root.End();
	FVActTextEmitUtils::EmitPropertyOpen(PropertyNames[(int32)_EPropName::Metas], Root.Cursors, Root.Source);
	const TCHAR* MetasBegin = Root.End();
	FVActTextEmitUtils::EmitArrayOpen(Root.Cursors, Root.Source);
	const int32 MetaCount = 0;//InContext->Objects.Num();
	for (int32 Index = 0; Index < MetaCount; ++Index)
	{
		const FOICMeta& Meta = InContext->Metas[Index];
		//  TODO Needs Meta Pass
		if (Index < MetaCount - 1) { FVActTextEmitUtils::EmitDelimiter(Root.Cursors, Root.Source); }
	}
	FVActTextEmitUtils::EmitArray(Root.Cursors, Root.Source, MetasBegin);
	FVActTextEmitUtils::EmitProperty(Root.Cursors, Root.Source, MetasPropBegin);

	FVActTextEmitUtils::EmitStruct(Root.Cursors, Root.Source, OICBegin);

	return true;
}

#undef LOCTEXT_NAMESPACE