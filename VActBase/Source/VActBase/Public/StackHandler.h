#pragma once

#include "IterateHandler.h"

#include "CoreMinimal.h"
#include "Containers/ContainerAllocationPolicies.h"
#include "Math/UnrealMathUtility.h"

template<typename InElementType, typename InAllocatorType>
class TStackSwapHandler;

template<typename InElementType, typename InAllocatorType>
class TStackHandler;

template<typename InAllocatorType = FDefaultAllocator>
struct VACTBASE_API TStackHandlerCursor
{
private:
	template<typename OtherInAllocatorType>
	friend struct TStackHandlerCursor;

public:
	typedef typename InAllocatorType::SizeType SizeType;

	SizeType Count;

	SizeType First;

	SizeType Stride;

	TStackHandlerCursor()
	{
		Stride = 1;
		Reset();
	}

	FORCEINLINE SizeType Num() const
	{
		return Count;
	}

	FORCEINLINE void Reset()
	{
		Count = 0;
		First = 0;
	}

	FORCEINLINE void Reverse(bool bComplex = true)
	{
		// TODO implement complex reverse, adjust for Count so all still points to corret range;
		Stride = -Stride;
	}

	FORCEINLINE bool IsEmpty() const
	{
		return Count <= 0;
	}
};

using FStackHandlerCursor = TStackHandlerCursor<FDefaultAllocator>;

template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
class VACTBASE_API TStackHandler
{
	template<typename OtherInElementType, typename OtherAllocator>
	friend class TStackHandler;

public:
	typedef TStackHandlerCursor<typename InAllocatorType> CursorType;
	typedef typename InAllocatorType::SizeType SizeType;
	typedef InElementType ElementType;
	typedef TArray<typename InElementType, typename InAllocatorType> ArrayType;
	typedef TIterateHandler<TStackHandler<typename InElementType, typename InAllocatorType>> IteratorType;

private:
	CursorType _Cursor;

	ArrayType* _ArrayPtr;

public:
	TStackHandler() : TStackHandler(nullptr)
	{
	}

	TStackHandler(ArrayType* ArrayPtr) : _ArrayPtr(ArrayPtr)
	{
		Empty();
	}

	TStackHandler(ArrayType& Array) : TStackHandler(&Array)
	{

	}

	~TStackHandler()
	{
		_ArrayPtr = nullptr;
	}

	/*FORCEINLINE*/ SizeType Push(ElementType& Element)
	{
		// TODO Craete Cyclic version and ensure both index and count;
		SizeType Index = _Cycle(_Cursor.First + _Cursor.Count);
		++_Cursor.Count;
		if (Index == _Cursor.First)
		{
			--_Cursor.Count;
			_Cursor.First = _Cycle(_Cursor.First + _Cursor.Stride);
		}
		(*_ArrayPtr)[Index] = Element;
		return Index;
	}

	FORCEINLINE SizeType Push(ElementType& Element, SizeType& Index)
	{
		Index = Push(Element);
		return Index;
	}

	FORCEINLINE ElementType& Peek()
	{
		SizeType Index;
		return Peek(Index);
	}

	/*FORCEINLINE*/ ElementType& Peek(SizeType& Index)
	{
		// Make sure it crashes if stack is empty
		Index = -1;
		if (_Cursor.Count > 0)
		{
			Index = _Cycle(_Cursor.First + _Cursor.Count - _Cursor.Stride);
		}
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE ElementType& Peek(SizeType& Index, ElementType& Into)
	{
		Into = Peek(Index);
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE ElementType& Pop()
	{
		SizeType Index;
		return Pop(Index);
	}

	/*FORCEINLINE*/ ElementType& Pop(SizeType& Index)
	{
		// Make sure it crashes if stack is empty
		Index = -1;
		if (_Cursor.Count > 0)
		{
			Index = _Cycle(_Cursor.First + _Cursor.Count - _Cursor.Stride);
			--_Cursor.Count;
		}
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE ElementType& Pop(SizeType& Index, ElementType& Into)
	{
		Into = Pop(Index);
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE void PopNull(ElementType NullDefault)
	{
		int32 Index;
		Pop(Index);
		(*_ArrayPtr)[Index] = NullDefault;
	}

	FORCEINLINE void PopNull(ElementType& Into, ElementType NullDefault)
	{
		int32 Index;
		Into = Pop(Index);
		(*_ArrayPtr)[Index] = NullDefault;
	}

	FORCEINLINE void PopNull(SizeType& Index, ElementType& Into, ElementType NullDefault)
	{
		Into = Pop(Index);
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

	FORCEINLINE void Reverse(const bool bComplex = true)
	{
		_Cursor.Reverse(bComplex);
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
			Index = _Cycle(_Cursor.First + (_Cursor.Stride)*Tracker);
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


template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
class VACTBASE_API TStackSwapHandler
{
	template<typename OtherInElementType, typename OtherAllocator>
	friend class TStackSwapHandler;

	template<typename OtherInElementType, typename OtherAllocator>
	friend class TStackHandler;

public:
	typedef TStackHandler<typename InElementType, typename InAllocatorType> HandlerType;
	typedef TStackHandlerCursor<typename InAllocatorType> CursorType;
	typedef typename InAllocatorType::SizeType SizeType;
	typedef InElementType ElementType;
	typedef TArray<typename InElementType, typename InAllocatorType> ArrayType;
	typedef TIterateHandler<TStackSwapHandler<typename InElementType, typename InAllocatorType>> IteratorType;

private:
	HandlerType Handlers[2];

	SizeType Pivot, PivotSwap;

public:
	TStackSwapHandler() : TStackSwapHandler(nullptr, nullptr)
	{
	}

	TStackSwapHandler(ArrayType* ArrayPtr) : TStackSwapHandler(ArrayPtr, ArrayPtr)
	{
	}

	TStackSwapHandler(ArrayType& Array) : TStackSwapHandler(&Array, &Array)
	{
	}

	TStackSwapHandler(ArrayType* ArrayPtr0, ArrayType* ArrayPtr1)
	{
		Pivot = 0;
		PivotSwap = 1;
		Handlers[Pivot] = HandlerType(ArrayPtr0);
		Handlers[PivotSwap] = HandlerType(ArrayPtr1);
		Handlers[PivotSwap].Reverse(false);
	}

	TStackSwapHandler(ArrayType& Array0, ArrayType& Array1) : TStackSwapHandler(&Array0, &Array1)
	{
	}

	~TStackSwapHandler()
	{
		Empty();
		Pivot = -1;
		PivotSwap = -1;
	}

	FORCEINLINE ElementType& Peek()
	{
		return Handlers[Pivot].Peek();
	}

	FORCEINLINE ElementType& Peek(SizeType& Index)
	{
		return Handlers[Pivot].Peek(Index);
	}

	FORCEINLINE ElementType& Peek(SizeType& Index, ElementType& Into)
	{
		return Handlers[Pivot].Peek(Index, Into);
	}

	FORCEINLINE ElementType& Pop()
	{
		return Handlers[Pivot].Pop();
	}

	FORCEINLINE void PopNull(ElementType NullDefault)
	{
		Handlers[Pivot].PopNull(NullDefault);
	}

	FORCEINLINE ElementType& PopBack()
	{
		ElementType Into;
		Handlers[Pivot].Pop(Into);
		Handlers[PivotSwap].Push(Into);
		return Into;
	}

	FORCEINLINE void PopBackNull(ElementType NullDefault)
	{
		ElementType Into;
		Handlers[Pivot].PopNull(Into, NullDefault);
		Handlers[PivotSwap].Push(Into);
	}

	FORCEINLINE SizeType PushBack(ElementType& Element)
	{
		return Handlers[PivotSwap].Push(Element);
	}

	FORCEINLINE SizeType PushBack(ElementType& Element, SizeType& Index)
	{
		return Handlers[PivotSwap].Push(Element, Index);
	}

	FORCEINLINE ElementType& Get(SizeType Index)
	{
		return Handlers[Pivot].Get(Index);
	}

	FORCEINLINE const ElementType& Get(SizeType Index) const
	{
		return Handlers[Pivot].Get(Index);
	}

	FORCEINLINE ArrayType& Array()
	{
		return Handlers[Pivot].Array();
	}

	FORCEINLINE const ArrayType& Array() const
	{
		return Handlers[Pivot].Array();
	}

	FORCEINLINE bool IsEmpty() const
	{
		return Handlers[Pivot].IsEmpty();
	}

	FORCEINLINE bool IsFull() const
	{
		return Handlers[PivotSwap].IsFull();
	}

	FORCEINLINE SizeType Num() const
	{
		return Handlers[Pivot].Num();
	}

	FORCEINLINE SizeType Max() const
	{
		return Handlers[PivotSwap].Max();
	}

	FORCEINLINE bool IsValid() const
	{
		return Handlers[Pivot].IsValid()
			&& Handlers[PivotSwap].IsValid();
	}

	FORCEINLINE void Empty()
	{
		Pivot = 0;
		PivotSwap = 1;
		Handlers[Pivot].Empty();
		Handlers[PivotSwap].Empty();
	}

	void Swap()
	{
		CursorType& Cursor = Handlers[Pivot]._Cursor;
		SizeType First = Cursor.First;
		const bool bNotEmpty = Cursor.Count > 0;

		Pivot = (Pivot + 1) % 2;
		PivotSwap = (Pivot + 1) % 2;
		CursorType& CursorSwap = Handlers[Pivot]._Cursor;

		CursorSwap.First = _Cycle(First + bNotEmpty * CursorSwap.Stride);
	}

	FORCEINLINE SizeType _Cycle(SizeType Index) const
	{
		const SizeType End = Handlers[Pivot]._Cycle(Index);
		return (Index < 0) * End + (Index % End);
	}

	bool _Next(SizeType& Counter, SizeType& Tracker, SizeType& Index) const
	{
		return Handlers[Pivot]._Next(Counter, Tracker, Index);
	}

	IteratorType Iterator() const
	{
		return IteratorType(this);
	}

};