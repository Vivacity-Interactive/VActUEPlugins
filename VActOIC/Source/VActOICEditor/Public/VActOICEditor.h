#pragma once

#include "UObject/Package.h"
#include "OICProfile.h"

#include "Components/SceneComponent.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "VActOICEditor.generated.h"

enum class ECmdVActExportOptions
{
	None = 0,
	PerLevel = 1,
	PerWorld = 2,
	AllCombined = 3,
};

//struct FSceneComponentIteratorCursor
//{
//	int32 Index;
//	USceneComponent* Parent;
//	const TArray<TObjectPtr<USceneComponent>>* Components;
//
//	explicit FSceneComponentIteratorCursor() = default;
//
//	explicit FSceneComponentIteratorCursor(const FSceneComponentIteratorCursor& Other) = default;
//
//	explicit FSceneComponentIteratorCursor(FSceneComponentIteratorCursor&& Other) = default;
//
//	explicit FSceneComponentIteratorCursor(USceneComponent* InParent)
//		: FSceneComponentIteratorCursor(InParent, 0)
//	{
//	}
//
//	explicit FSceneComponentIteratorCursor(USceneComponent* InParent, int32 InIndex)
//		: Index(InIndex)
//		, Parent(InParent)
//		, Components(&InParent->GetAttachChildren())
//	{
//	}
//
//	FORCEINLINE USceneComponent* operator*() const
//	{
//		return (*Components)[Index];
//	}
//
//	FORCEINLINE USceneComponent* operator->() const
//	{
//		return (*Components)[Index];
//	}
//
//	FORCEINLINE FSceneComponentIteratorCursor& operator++()
//	{
//		++Index;
//		return *this;
//	}
//
//	FORCEINLINE FSceneComponentIteratorCursor& operator--()
//	{
//		--Index;
//		return *this;
//	}
//
//	FORCEINLINE FSceneComponentIteratorCursor& operator+=(int32 Offset)
//	{
//		Index += Offset;
//		return *this;
//	}
//
//	FORCEINLINE FSceneComponentIteratorCursor& operator-=(int32 Offset)
//	{
//		Index -= Offset;
//		return *this;
//	}
//
//	FORCEINLINE explicit operator bool() const
//	{
//		return Parent && Components && Components->IsValidIndex(Index);
//	}
//
//	FSceneComponentIteratorCursor& operator=(FSceneComponentIteratorCursor&& Other) = default;
//
//	FSceneComponentIteratorCursor& operator=(const FSceneComponentIteratorCursor& Other) = default;
//
//};

////template<typename ComponentType>
//struct TSceneComponentIterator
//{
//	FSceneComponentIteratorCursor Cursor;
//	TArray<USceneComponent*> Stack;
//
//	explicit TSceneComponentIterator(AActor* InActor)
//	{
//		TInlineComponentArray<USceneComponent*> Components(InActor);
//		for (USceneComponent* Component : Components)
//		{
//			Stack.Push(Component);
//			UE_LOG(LogTemp, Warning, TEXT("%s Add %s"), *GetNameSafe(InActor), *GetNameSafe(Component));
//		}
//
//		if (!Stack.IsEmpty())
//		{
//			Cursor = FSceneComponentIteratorCursor(Stack.Pop());
//			//UE_LOG(LogTemp, Warning, TEXT("Current %s"), *GetNameSafe(*Cursor));
//			_Prepare();
//		}
//	}
//
//	explicit TSceneComponentIterator(USceneComponent* InRoot)
//		: Cursor(InRoot)
//	{
//		_Prepare();
//	}
//
//	void _Prepare()
//	{
//		const bool bPush = !!Cursor && Cursor->GetNumChildrenComponents() > 0;
//		if (bPush)
//		{
//			Stack.Push(*Cursor);
//		}
//	}
//	
//	void operator++()
//	{
//		++Cursor;
//		const bool bPop = !Cursor && !Stack.IsEmpty();
//		if (bPop)
//		{
//			Cursor = FSceneComponentIteratorCursor(Stack.Pop());
//		}
//		_Prepare();
//	}
//
//	FORCEINLINE USceneComponent* operator*() const
//	{
//		return *Cursor;
//	}
//
//	FORCEINLINE USceneComponent* operator->() const
//	{
//		return *Cursor;
//	}
//
//	FORCEINLINE explicit operator bool() const
//	{ 
//		return !!Cursor;
//	}
//
//};

USTRUCT()
struct VACTOICEDITOR_API FVActOICEditor
{
	GENERATED_BODY()

	static const TMap<FString, ECmdVActExportOptions> ExportOptionNameToEnum;

public:
	static const TArray<FString> ExportOptionNames;

	static void Cmd_ExportToOICAsset(const TArray<FString>& Args);

	static void ExportToOICAsset(const UWorld* World, const FString& Path, ECmdVActExportOptions Options = ECmdVActExportOptions::AllCombined);

protected:
	static void _ResolveSelected(TArray<FAssetData>& Assets);

	static void _SaveOICProfileAsset(UOICProfile* Profile, UPackage* Package);

	static void _ResolveOIC(UOICProfile* Profile, UPackage* Package);

};