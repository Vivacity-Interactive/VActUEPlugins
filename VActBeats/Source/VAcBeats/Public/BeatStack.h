#pragma once

#include "VActBeatsTypes.h"

#include "BeatContext.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "BeatStack.generated.h"

UCLASS(editinlinenew, BlueprintType, Blueprintable)
class VACTBEATS_API UBeatStack : public UObject
{
	GENERATED_BODY()

	TMap<FName, int32> NameToId;

	TMap<int32, FBeatClaim> IdToClaim;

public:
#if WITH_EDITORONLY_DATA
	//UPROPERTY()
	//FVActFileIOInfo IOInfo;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 ContextCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 PointCount;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (MultiLine = "true"))
	FString Notes;
#endif

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Title;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName Name;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<FBeatEntity> Entities;

	UPROPERTY()
	TArray<FBeatContext> Contexts;

	UPROPERTY()
	TArray<FBeatPointPrototype> Points;

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

	void Init();

#if WITH_EDITORONLY_DATA
	// Todo Create Edit Context or Point Entry here
	/*
	//UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Entity")
	//int32 EditEntityId;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Entity")
	//FName EditEntityName;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Entity")
	//FBeatEntity Entity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Context")
	int32 EditContextId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Context")
	FBeatPointPrototype Context;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Point")
	int32 EditPointId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Edit Beat Point")
	FBeatPointPrototype Point;

	UFUNCTION(CallInEditor, Category = "Edit Beat Entity")
	SaveEntity();

	UFUNCTION(CallInEditor, Category = "Edit Beat Context")
	SaveContext();

	UFUNCTION(CallInEditor, Category = "Edit Beat Point")
	SavePoint();	
	*/
#endif

};