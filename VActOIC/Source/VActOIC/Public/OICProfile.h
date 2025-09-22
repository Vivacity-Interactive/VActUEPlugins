#pragma once

#if WITH_EDITORONLY_DATA
#include "VActFileEntryTypes.h"
#endif

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "OICProfile.generated.h"

UENUM()
enum class EOICShape
{
	None = 0,
	Box,
	Sphere,
	Capsule,
	Convex,
};

UENUM()
enum class EOICAsset
{
	None = 0,
	Actor,
	Mesh,
	Particle,
	Data,
	Audio,
	System,
	Collider,
};

UENUM()
enum class EOICValue
{
	None = 0,
	Name,
	String,
	Bool,
	Float,
	Float2,
	Float3,
	Float4,
	Float5,
	Float6,
	Floats,
	Int,
	Int2,
	Int3,
	Int4,
	Int5,
	Int6,
	Ints,
	Names,
	Strings,
	Asset
};

USTRUCT()
struct VACTOIC_API FOICTracker
{
	GENERATED_BODY()

	UPROPERTY()
	TWeakObjectPtr<AActor> Actor;

	UPROPERTY()
	TWeakObjectPtr<USceneComponent> Component;

	// Maybe set this as a UObject instead

	UPROPERTY()
	int32 Index;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTOIC_API FOICProfileEntry
{
	GENERATED_BODY()

	UPROPERTY()
	TMap<FName, FOICTracker> Trackers;

	TMap<FName, TWeakObjectPtr<UInstancedStaticMeshComponent>> ISMCs;

	TMap<FName, TWeakObjectPtr<AActor>> Colliders;

	TMap<FName, TObjectPtr<UObject>> Datas;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bSkip : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bTracked : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bInitialized : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bUpdateMetas : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bUseStaticMeshInstances : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bClear : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bUseColliderShapePrefix : 1;

	//UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	//TObjectPtr<UOICFormatProfile> Format;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TSubclassOf<UOICProfile> Class;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TObjectPtr<UOICProfile> Object;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TObjectPtr<AActor> Parent;

	FOICProfileEntry();
};

struct VACTOIC_API FOICValue
{
	static const TArray<FName> ValueTypes;

	static const TMap<FName, EOICValue> MapValueType;

	uint8 Flag;

	EOICValue Type;

	union
	{
		struct
		{
			alignas(FString) uint8 _Asset[sizeof(FString)];
			EOICAsset _Type;
		};
		alignas(FName) uint8 _Name[sizeof(FName)];
		alignas(FString) uint8 _String[sizeof(FString)];
		alignas(TArray<float>) uint8 _Floats[sizeof(TArray<float>)];
		alignas(TArray<int32>) uint8 _Ints[sizeof(TArray<int32>)];
		alignas(TArray<FName>) uint8 _Names[sizeof(TArray<FName>)];
		alignas(TArray<FString>) uint8 _Strings[sizeof(TArray<FString>)];
		uint8 _Raw[24];
		float _FData[6];
		int32 _IData[6];
	};

	FOICValue() : Flag(0), Type(EOICValue::None), _Raw{ 0 } { }

	FOICValue(const FOICValue& Other) noexcept { _Copy(*this, Other); }

	FOICValue(FOICValue&& Other) noexcept { _Move(*this, Other); }

	~FOICValue() { Reset(); }

	FOICValue& operator=(const FOICValue& Other) noexcept { Reset(); _Copy(*this, Other); return *this; }

	FOICValue& operator=(FOICValue&& Other) noexcept { Reset(); _Move(*this, Other); return *this; }

	template<typename T>
	T& GetValue()
	{
		check(sizeof(T) <= sizeof(_Raw));
		return *reinterpret_cast<T*>(_Raw);
	}

	template<typename T>
	const T& GetValue() const
	{
		check(sizeof(T) <= sizeof(_Raw));
		return *reinterpret_cast<const T*>(_Raw);
	}

	void Reset()
	{
		if (Flag)
		{
			switch (Type)
			{
			case EOICValue::Name: reinterpret_cast<FName*>(_Name)->~FName(); break;
			case EOICValue::String: reinterpret_cast<FString*>(_String)->~FString(); break;
			case EOICValue::Floats: reinterpret_cast<TArray<float>*>(_Floats)->~TArray<float>(); break;
			case EOICValue::Ints: reinterpret_cast<TArray<int32>*>(_Ints)->~TArray<int32>(); break;
			case EOICValue::Names: reinterpret_cast<TArray<FName>*>(_Names)->~TArray<FName>(); break;
			case EOICValue::Strings: reinterpret_cast<TArray<FString>*>(_Strings)->~TArray<FString>(); break;
			}
		}
		Flag = 0;
		Type = EOICValue::None;
	}

	static void _Move(FOICValue& Lhs, FOICValue& Rhs)
	{
		if (&Lhs != &Rhs)
		{
			Lhs.Flag = 0;
			Lhs.Type = Rhs.Type;
			switch (Rhs.Type)
			{
			case EOICValue::Name: new (Lhs._Name) FName(MoveTemp(*reinterpret_cast<FName*>(Rhs._Name))); Lhs.Flag = 1; break;
			case EOICValue::String: new (Lhs._String) FString(MoveTemp(*reinterpret_cast<FString*>(Rhs._String))); Lhs.Flag = 1; break;
			case EOICValue::Floats: new (Lhs._Floats) TArray<float>(MoveTemp(*reinterpret_cast<TArray<float>*>(Rhs._Floats))); Lhs.Flag = 1; break;
			case EOICValue::Ints: new (Lhs._Ints) TArray<int32>(MoveTemp(*reinterpret_cast<TArray<int32>*>(Rhs._Ints))); Lhs.Flag = 1; break;
			case EOICValue::Names: new (Lhs._Names) TArray<FName>(MoveTemp(*reinterpret_cast<TArray<FName>*>(Rhs._Names))); Lhs.Flag = 1; break;
			case EOICValue::Strings: new (Lhs._Strings) TArray<FString>(MoveTemp(*reinterpret_cast<TArray<FString>*>(Rhs._Strings))); Lhs.Flag = 1; break;
			default: FMemory::Memmove(Lhs._Raw, Rhs._Raw, sizeof(Lhs._Raw)); break;
			}
			Rhs.Reset();
		}
	}

	static void _Copy(FOICValue& Lhs, const FOICValue& Rhs)
	{
		if (&Lhs != &Rhs)
		{
			Lhs.Flag = 0;
			Lhs.Type = Rhs.Type;
			switch (Rhs.Type)
			{
			case EOICValue::Name: new (Lhs._Name) FName(CopyTemp(*reinterpret_cast<const FName*>(Rhs._Name))); Lhs.Flag = 1; break;
			case EOICValue::String: new (Lhs._String) FString(CopyTemp(*reinterpret_cast<const FString*>(Rhs._String))); Lhs.Flag = 1; break;
			case EOICValue::Floats: new (Lhs._Floats) TArray<float>(CopyTemp(*reinterpret_cast<const TArray<float>*>(Rhs._Floats))); Lhs.Flag = 1; break;
			case EOICValue::Ints: new (Lhs._Ints) TArray<int32>(CopyTemp(*reinterpret_cast<const TArray<int32>*>(Rhs._Ints))); Lhs.Flag = 1; break;
			case EOICValue::Names: new (Lhs._Names) TArray<FName>(CopyTemp(*reinterpret_cast<const TArray<FName>*>(Rhs._Names))); Lhs.Flag = 1; break;
			case EOICValue::Strings: new (Lhs._Strings) TArray<FString>(CopyTemp(*reinterpret_cast<const TArray<FString>*>(Rhs._Strings))); Lhs.Flag = 1; break;
			default: FMemory::Memcpy(Lhs._Raw, Rhs._Raw); break;
			}
		}
	}
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTOIC_API FOICMetaEntry
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TSubclassOf<UActorComponent> Asset;

	TMap<FName, FOICValue> Properties;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTOIC_API FOICMeta
{
	GENERATED_BODY()
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FOICMetaEntry> Entries;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTOIC_API FOICObject
{
	GENERATED_BODY()

	static const TArray<FName> AssetTypes;

	static const TMap<FName, EOICAsset> MapAssetType;

	static const TArray<FName> ShapeTypes;

	static const TMap<FName, EOICShape> MapShapeType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	EOICAsset Type;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (EditCondition = "Type==EOICAsset::Collider", EditConditionHides))
	EOICShape Shape;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (EditCondition = "Type==EOICAsset::Mesh || Type==EOICAsset::Particle || Type==EOICAsset::Collider", EditConditionHides))
	TObjectPtr<UStaticMesh> Mesh;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta  = (EditCondition = "Type==EOICAsset::Actor", EditConditionHides))
	TSubclassOf<AActor> Actor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (EditCondition = "Type==EOICAsset::Data || Type==EOICAsset::System", EditConditionHides))
	TSubclassOf<UObject> Data;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (EditCondition = "Type==EOICAsset::Audio", EditConditionHides))
	TObjectPtr<USoundWave> Audio;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Meta;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTOIC_API FOICInstance
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Id;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Object;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Parent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Meta;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FTransform Transform;
};

UCLASS(Blueprintable, BlueprintType)
class VACTOIC_API UOICProfile : public UObject
{
	GENERATED_BODY()

public:
#if WITH_EDITORONLY_DATA
	UPROPERTY()
	FVActFileIOInfo IOInfo;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 InstancesCount;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (MultiLine = "true"))
	FString Notes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;
#endif

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Type;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Version;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Axis;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Name;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FOICObject> Objects;

	UPROPERTY()
	TArray<FOICInstance> Instances;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FOICMeta> Metas;
};
