#pragma once

#include "VActMathTypes.h"
#include "VActBeatsTypes.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "BeatsProfile.generated.h"

UCLASS(Blueprintable, BlueprintType)
class VACTBEATS_API UBeatsProfile : public UObject
{
	static const TArray<FName> ModeTypes;

	static const TMap<FName, EBeatMode> MapModeTypes;

	GENERATED_BODY()

private:
	TMap<FName, int32> NameToId;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (EditFixedSize, AllowPrivateAccess = "true"))
	TMap<int32, FBeatClaim> IdToClaim;

	TVectorBuffer<float> Points;

	TVectorBuffer<float> Vectors;

	TVectorBuffer<float> Intervals;

	TVectorBuffer<EMathOperation> Effectors;

	TStructBuffer<FBeatEntity> Entities;

	TStructBuffer<FBeatContext> Contexts;

	TStructBuffer<FBeatPrototype> Prototypes;

public:
#if WITH_EDITORONLY_DATA
	//UPROPERTY()
	//FVActFileIOInfo IOInfo;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 PointCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 VectorCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 IntervalCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 EffectorCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 PrototypeCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 ContextCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 VectorSize;

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
	EBeatMode Mode;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FName> Header;

public:
	FORCEINLINE int32 ResolveId(FName ByName)
	{
		int32* IdPtr = NameToId.Find(ByName);
		return IdPtr ? (*IdPtr) : -1;
	}

	FORCEINLINE bool HasEntity(int32 ById, FBeatClaim)
	{
		return IdToClaim.Contains(ById);
	}

	FORCEINLINE bool HasInteraction(int32 Self, int32 Target)
	{
		return IdToClaim.Contains(Self) && IdToClaim.Contains(Target);
	}

	FORCEINLINE FBeatClaim* ClaimEntity(int32 ById, bool bNearestUnclaimedMatch = false)
	{
		return IdToClaim.Find(ById);
	}

	template<uint32 NumFeatures>
	FORCEINLINE bool IsValidVector(const TBeatVector<NumFeatures>& Vector)
	{
		return Header.Num() == NumFeatures;
	}

	template<typename T0>
	FORCEINLINE bool IsValidVector(const TBeatVectorOf<T0>& Vector)
	{
		return Header.Num() == Vector.Num();
	}

	template<uint32 NumFeatures>
	FORCEINLINE bool IsValidEffector(const TBeatEffector<NumFeatures>& Effector)
	{
		return Header.Num() == NumFeatures;
	}

	template<typename T0>
	FORCEINLINE bool IsValidEffector(const TBeatEffectorOf<T0>& Effector)
	{
		return Header.Num() == Effector.Num();
	}

	void InitClaims(bool bClear = true);
};
