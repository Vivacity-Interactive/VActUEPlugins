#pragma once

#include "VActMathTypes.h"

#include "CoreMinimal.h"
#include "VActBeatsTypes.generated.h"

template<uint32 NumFeatures>
struct TBeatVector : TMathVector<float, NumFeatures>
{
};

template<uint32 NumFeatures>
struct TBeatEffector : TMathEffector<NumFeatures>
{
};

template<uint32 NumFeatures>
struct TBeatEffect : TMathEffect<float, NumFeatures>
{
};

template<typename TypeBeatVector>
using TBeatVectorOf = TBeatVector<sizeof(TypeBeatVector) / sizeof(float)>;

template<typename TypeBeatVector>
using TBeatEffectorOf = TBeatEffector<sizeof(TypeBeatVector) / sizeof(float)>;

template<typename TypeBeatVector>
using TBeatEffectOf = TBeatEffector<sizeof(TypeBeatVector) / sizeof(float)>;


USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatEntity
{
	GENERATED_BODY()

#if WITH_EDITORONLY_DATA
	UPROPERTY(EditAnywhere, Category = Prototype, meta = (MultiLine = "true"))
	FString Notes;
#endif

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 Id;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	FName Name;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;

	FBeatEntity();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatClaim
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatEntity Entity;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	AActor* Actor;

	/*UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UBeatContextComponent* Component;*/

	FBeatClaim();

	FBeatClaim(const FBeatEntity& InEntity, AActor* InActor = nullptr);
};


USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatMeta
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Name;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Self;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Context;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Class;

	FBeatMeta();

	FBeatMeta(const FBeatMeta&) = default;

	FBeatMeta(FBeatMeta&&) = default;
	
	FBeatMeta& operator=(const FBeatMeta&) = default;
	
	FBeatMeta& operator=(FBeatMeta&&) = default;
};


USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateTime
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Causality;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Branch;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Correlation;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Used;

	FBeatStateTime() = default;

	FBeatStateTime(const FBeatStateTime&) = default;

	FBeatStateTime(FBeatStateTime&&) = default;

	FBeatStateTime& operator=(const FBeatStateTime&) = default;

	FBeatStateTime& operator=(FBeatStateTime&&) = default;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateTimeMinimal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Causality;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Used;

	FBeatStateTimeMinimal() = default;

	FBeatStateTimeMinimal(const FBeatStateTimeMinimal&) = default;

	FBeatStateTimeMinimal(FBeatStateTimeMinimal&&) = default;

	FBeatStateTimeMinimal& operator=(const FBeatStateTimeMinimal&) = default;

	FBeatStateTimeMinimal& operator=(FBeatStateTimeMinimal&&) = default;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateExternal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Interrupt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Distance;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Facing;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Direction;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Approach;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Focus;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float BAC;

	FBeatStateExternal() = default;

	FBeatStateExternal(const FBeatStateExternal&) = default;

	FBeatStateExternal(FBeatStateExternal&&) = default;

	FBeatStateExternal& operator=(const FBeatStateExternal&) = default;

	FBeatStateExternal& operator=(FBeatStateExternal&&) = default;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateExternalMinimal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Interrupt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Distance;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Focus;

	FBeatStateExternalMinimal() = default;

	FBeatStateExternalMinimal(const FBeatStateExternalMinimal&) = default;

	FBeatStateExternalMinimal(FBeatStateExternalMinimal&&) = default;

	FBeatStateExternalMinimal& operator=(const FBeatStateExternalMinimal&) = default;

	FBeatStateExternalMinimal& operator=(FBeatStateExternalMinimal&&) = default;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateInternal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Shame;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Resentment;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Anxious;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Excitement;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Confidence;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Comfort;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Arousal;

	FBeatStateInternal() = default;

	FBeatStateInternal(const FBeatStateInternal&) = default;

	FBeatStateInternal(FBeatStateInternal&&) = default;

	FBeatStateInternal& operator=(const FBeatStateInternal&) = default;

	FBeatStateInternal& operator=(FBeatStateInternal&&) = default;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateInternalMinimal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Aware;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Comfort;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Secure;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Trusting;

	FBeatStateInternalMinimal() = default;

	FBeatStateInternalMinimal(const FBeatStateInternalMinimal&) = default;

	FBeatStateInternalMinimal(FBeatStateInternalMinimal&&) = default;

	FBeatStateInternalMinimal& operator=(const FBeatStateInternalMinimal&) = default;

	FBeatStateInternalMinimal& operator=(FBeatStateInternalMinimal&&) = default;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatStateReflection
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Mindfool;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Empathy;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Sympathy;

	FBeatStateReflection() = default;

	FBeatStateReflection(const FBeatStateReflection&) = default;

	FBeatStateReflection(FBeatStateReflection&&) = default;

	FBeatStateReflection& operator=(const FBeatStateReflection&) = default;

	FBeatStateReflection& operator=(FBeatStateReflection&&) = default;
};


USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatVector
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateTime Time;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateExternal External;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateInternal Internal;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateReflection Reflection;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float _Noise;

	FBeatVector() = default;

	FBeatVector(const FBeatVector&) = default;

	FBeatVector(FBeatVector&&) = default;

	FBeatVector& operator=(const FBeatVector&) = default;

	FBeatVector& operator=(FBeatVector&&) = default;

	FORCEINLINE operator float* ()
	{
		return reinterpret_cast<float*>(this);
	}

	FORCEINLINE operator const float* () const
	{
		return reinterpret_cast<const float*>(this);
	}
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatVectorMinimal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateTime Time;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateExternalMinimal External;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatStateInternalMinimal Internal;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float _Noise;

	FBeatVectorMinimal() = default;

	FBeatVectorMinimal(const FBeatVectorMinimal&) = default;

	FBeatVectorMinimal(FBeatVectorMinimal&&) = default;

	FBeatVectorMinimal& operator=(const FBeatVectorMinimal&) = default;

	FBeatVectorMinimal& operator=(FBeatVectorMinimal&&) = default;

	FORCEINLINE operator float* ()
	{
		return reinterpret_cast<float*>(this);
	}

	FORCEINLINE operator const float* () const
	{
		return reinterpret_cast<const float*>(this);
	}
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatPointPrototype
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatMeta Meta;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Weight;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatVector Point;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FWeightedEntry> Contexts;

};

USTRUCT(BlueprintType, Blueprintable)
struct VACTBEATS_API FBeatPointPrototypeMinimal
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatMeta Meta;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Weight;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FBeatVectorMinimal Point;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FWeightedEntry> Contexts;

};


//USTRUCT(BlueprintType, Blueprintable)
//struct VACTBEATS_API FBeatIntervalPrototype
//{
//	GENERATED_BODY()
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	FBeatMeta Meta;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	float Weight;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	FBeatVector Min;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	FBeatVector Max;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	TArray<FWeightedEntry> Contexts;
//
//};
//
//USTRUCT(BlueprintType, Blueprintable)
//struct VACTBEATS_API FBeatIntervalPrototypeMinimal
//{
//	GENERATED_BODY()
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	FBeatMeta Meta;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	float Weight;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	FBeatVectorMinimal Min;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	FBeatVectorMinimal Max;
//
//	UPROPERTY(EditAnywhere, BlueprintReadWrite)
//	TArray<FWeightedEntry> Contexts;
//
//};