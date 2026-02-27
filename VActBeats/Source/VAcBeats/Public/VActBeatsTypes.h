#pragma once

#include "VActMathTypes.h"

#include "CoreMinimal.h"
#include "VActBeatsTypes.generated.h"

UENUM()
enum class EBeatAsset
{
	None = 0,
	Asset,
	Text
};

UENUM()
enum class EBeatMode
{
	None = 0,
	Scalar,
	Vector
	//Modulo,
	//Minimum,
};

UENUM()
enum class EBeatPrototype
{
	None = 0,
	Point,
	Vector,
	Interval
};

template<uint32 NumFeatures>
struct TBeatVector : TMathVector<float, NumFeatures>
{
};

template<uint32 NumFeatures>
struct TBeatEffector : TMathEffector<NumFeatures>
{
};

template<typename TypeBeatVector>
using TBeatVectorOf = TBeatVector<sizeof(TypeBeatVector) / sizeof(float)>;

template<typename TypeBeatVector>
using TBeatEffectorOf = TBeatEffector<sizeof(TypeBeatVector) / sizeof(float)>;

template<typename T0, typename T1 = uint8>
struct TAlignasObject {
	alignas(TSubclassOf<T0>) T1 Class[sizeof(TSubclassOf<T0>)];
	alignas(TObjectPtr<T0>) T1 Object[sizeof(TObjectPtr<T0>)];
};

struct VACTBEATS_API FBeatAsset
{
	static const TArray<FName> AssetTypes;

	static const TMap<FName, EBeatAsset> MapAssetTypes;

	EBeatAsset Type;

	FName Name;

	union
	{
		TAlignasObject<UObject> Asset;
		alignas(FString) uint8 Text[sizeof(FString)];
		uint8 _Raw[16];
		uint8* _Ptr;
	};

	FBeatAsset();

	template<typename T0>
	void FORCEINLINE PlaceAssetClass(TSubclassOf<T0> InClass)
	{
		new (Asset.Class) TSubclassOf<T0>(InClass);
	}

	template<typename T0>
	void FORCEINLINE PlaceAssetInstance(TSubclassOf<T0> InObject)
	{
		new (Asset.Object) TObjectPtr<T0>(InObject);
	}

	template<typename T0>
	void FORCEINLINE PlaceText(FString InText)
	{
		new (Text) FString(MoveTemp(InText));
	}
};

struct VACTBEATS_API FBeatEffect
{
	//int32 Size;

	//EBeatEffectMode Mode;

	float* Vector;

	EMathOperation* Effector;

	FBeatEffect();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatTime
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Time")
	float Causality;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Time")
	float Branch;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatVector|Time")
	float Correlation;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatEntity
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Id;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	FName Name;

	FBeatEntity();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatClaim
{
	GENERATED_BODY()

	const FBeatEntity* Entity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	AActor* Actor;

	/*UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UBeatContextComponent* Component;*/

	FBeatClaim();

	FBeatClaim(const FBeatEntity* InEntity, AActor* InActor = nullptr);
};


USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatMeta
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Self;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Context;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Class;

	FBeatMeta();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatContext
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Name;

	TArray<FBeatAsset> Assets;

	FBeatContext();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatPrototype
{
	static const TArray<FName> PrototypeTypes;

	static const TMap<FName, EBeatPrototype> MapPrototypeTypes;

	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Id;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatMeta Meta;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Weight;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	EBeatPrototype Type;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite)
	//int32 CoordinateId;

	float* Coordinate;

	FBeatEffect Effect;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite)
	//TArray<int32> Contexts;

	TArray<FBeatContext*> Contexts;

	FBeatPrototype();
};