#pragma once

#include "IterateHandler.h"

#include "CoreMinimal.h"
#include "Containers/ContainerAllocationPolicies.h"
#include "Math/UnrealMathUtility.h"

template<typename InAllocatorType = FDefaultAllocator>
struct VACTBASE_API TQueueHandlerCursor
{
private:
	template<typename OtherInAllocatorType>
	friend struct TQueueHandlerCursor;

public:
	typedef typename InAllocatorType::SizeType SizeType;

	SizeType First;

	SizeType Count;

	TQueueHandlerCursor()
	{
		Reset();
	}

	FORCEINLINE SizeType Num() const
	{
		return Count;
	}

	FORCEINLINE void Reset()
	{
		First = 0;
		Count = 0;
	}

	FORCEINLINE bool IsEmpty() const
	{
		return Count <= 0;
	}
};

using FQueueHandlerCursor = TQueueHandlerCursor<FDefaultAllocator>;

template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
class VACTBASE_API TQueueHandler
{
	template<typename OtherInElementType, typename OtherAllocator>
	friend class TQueueHandler;

public:
	typedef TQueueHandlerCursor<typename InAllocatorType> CursorType;
	typedef typename InAllocatorType::SizeType SizeType;
	typedef InElementType ElementType;
	typedef TArray<typename InElementType, typename InAllocatorType> ArrayType;
	typedef TIterateHandler<TQueueHandler<typename InElementType, typename InAllocatorType>> IteratorType;

private:
	CursorType _Cursor;

	ArrayType* _ArrayPtr;

public:
	TQueueHandler() : TQueueHandler(nullptr)
	{
	}

	TQueueHandler(ArrayType* ArrayPtr) : _ArrayPtr(ArrayPtr)
	{
		Empty();
	}

	TQueueHandler(ArrayType& Array) : TQueueHandler(&Array)
	{

	}

	~TQueueHandler()
	{
		_ArrayPtr = nullptr;
	}

	/*FORCEINLINE*/ SizeType Enqueue(ElementType& Element)
	{
		const bool bFull = _Cursor.Count >= (*_ArrayPtr).Num();
		if (bFull)
		{
			_Cursor.First = _Cycle(_Cursor.First + 1);
			--_Cursor.Count;
		}
		
		SizeType Index = _Cycle(_Cursor.First + _Cursor.Count);
		++_Cursor.Count;
		(*_ArrayPtr)[Index] = Element;
		return Index;
	}

	FORCEINLINE SizeType Enqueue(ElementType& Element, SizeType& Index)
	{
		Index = Add(Element);
		return Index;
	}

	FORCEINLINE ElementType& Dequeue()
	{
		SizeType Index;
		return Dequeue(Index);
	}

	/*FORCEINLINE*/ ElementType& Dequeue(SizeType& Index)
	{
		// Make sure it crashes if queue is empty
		Index = -1;

		const bool bNotEmpty = _Cursor.Count > 0;
		if (bNotEmpty)
		{
			Index = _Cursor.First;
			_Cursor.First = _Cycle(_Cursor.First + 1);
			--_Cursor.Count;
		}

		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE ElementType& Dequeue(SizeType& Index, ElementType& Into)
	{
		Into = Dequeue(Index);
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE void DequeueNull(ElementType NullDefault)
	{
		int32 Index;
		Dequeue(Index);
		(*_ArrayPtr)[Index] = NullDefault;
	}

	FORCEINLINE void DequeueNull(ElementType& Into, ElementType NullDefault)
	{
		int32 Index;
		Into = Dequeue(Index);
		(*_ArrayPtr)[Index] = NullDefault;
	}

	FORCEINLINE void DequeueNull(SizeType& Index, ElementType& Into, ElementType NullDefault)
	{
		Into = Dequeue(Index);
		(*_ArrayPtr)[Index] = NullDefault;
	}

	FORCEINLINE ElementType& Get(SizeType Index)
	{
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE const ElementType& Get(SizeType Index) const
	{
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE ArrayType& Array()
	{
		return (*_ArrayPtr);
	}

	FORCEINLINE const ArrayType& Array() const
	{
		return (*_ArrayPtr);
	}

	FORCEINLINE bool IsEmpty() const
	{
		return _Cursor.IsEmpty();
	}

	FORCEINLINE bool IsFull() const
	{
		return Num() >= Max();
	}

	FORCEINLINE void Empty()
	{
		_Cursor.Reset();
	}

	FORCEINLINE SizeType Num() const
	{
		return _Cursor.Num();
	}

	FORCEINLINE SizeType Max() const
	{
		return (*_ArrayPtr).Num();
	}

	FORCEINLINE bool IsValid() const
	{
		return _ArrayPtr != nullptr;
	}

	FORCEINLINE bool IsValid(SizeType Index) const
	{
		return IsValid() && 0 > Index && Index < (*_ArrayPtr).Num();
	}

	FORCEINLINE const CursorType& Cursor() const
	{
		return _Cursor;
	}

	FORCEINLINE SizeType _Cycle(SizeType Index) const
	{
		// TODO we can always assume -1 =< Index < 2 * _Array.Num()
		// this way we may avoid needing to use %
		const SizeType End = (*_ArrayPtr).Num();
		return (Index < 0) * End + (Index % End);
	}

	bool _Next(SizeType& Counter, SizeType& Tracker, SizeType& Index) const
	{
		Index = -1;
		bool bNext = Tracker >= 0 && Tracker < _Cursor.Count;
		if (bNext)
		{
			Index = _Cycle(_Cursor.First + Tracker);
			++Counter;
		}
		++Tracker;
		return bNext;
	}

	IteratorType Iterator() const
	{
		return IteratorType(this);
	}
};

