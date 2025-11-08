#pragma once

#include "CoreMinimal.h"
//#include "IterateHandler.h"

//template<typename ElementType, template<typename> class HandlerType>
//struct FIteratorHandler;

template<typename ElementType>
struct FBufferHandler;

template<typename ElementType>
struct VACTBASE_API FBufferHandler
{
	int32 Head;

	int32 Size;

	TArray<ElementType>* Array;

	FBufferHandler()
		: Head(0)
		, Size(0)
		, Array(nullptr)
	{
	}

	FBufferHandler(TArray<ElementType>& InArray)
		: Head(0)
		, Size(InArray.Num())
		, Array(&InArray)
	{
	}

	void Add(ElementType&& Item)
	{
		(*Array)[Head] = MoveTemp(Item);
		Head = ++Head % (*Array).Num();
		Size = FMath::Min(++Size, (*Array).Num());
	}

	void Add(ElementType& Item)
	{
		(*Array)[Head] = Item;
		Head = ++Head % (*Array).Num();
		Size = FMath::Min(++Size, (*Array).Num());
	}

	void Reset(bool bResetArray = false)
	{
		Head = 0;
		Size = 0;
		if (bResetArray) { (*Array).Reset(); }
	}

	template<typename Func>
	void ForEach(Func&& Callback) const
	{
		for (int32 Index = 0; Index < Size; ++Index)
		{
			const int32 _Index = BufferIndex(Index);
			Callback((*Array)[_Index]);
		}
	}

	constexpr FORCEINLINE int32 BufferIndex(int32 Index) const { return (Head + (*Array).Num() - Size + Index) % (*Array).Num(); }

	constexpr FORCEINLINE int32 Num() const { return Size; }

	FORCEINLINE ElementType& operator[](int32 Index)
	{
		const int32 _Index = BufferIndex(Index);
		return (*Array)[_Index];
	}

	FORCEINLINE const ElementType& operator[](int32 Index) const
	{
		const int32 _Index = BufferIndex(Index);
		return (*Array)[_Index];
	}

	/*FIteratorHandler<ElementType, FBufferHandler<ElementType>> Iterator() const
	{
		return FIteratorHandler(*this, 0);
	}

	/*FIteratorHandler<ElementType, FBufferHandler<ElementType>> begin()
	{
		return FIteratorHandler(*this, 0);
	}

	FIteratorHandler<ElementType, FBufferHandler<ElementType>> end()
	{
		return FIteratorHandler(*this, Size);
	}*/

};