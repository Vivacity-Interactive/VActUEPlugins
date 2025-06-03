// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "IterateHandler.h"

#include "CoreMinimal.h"
#include "Containers/ContainerAllocationPolicies.h"
#include "Math/UnrealMathUtility.h"



template<typename InAllocatorType = FDefaultAllocator>
struct VACTBASE_API TListHandlerCursor
{
private:
	template<typename OtherInAllocatorType>
	friend struct TListHandlerCursor;

public:
	typedef typename InAllocatorType::SizeType SizeType;
	typedef TArray<SizeType, typename InAllocatorType> ArrayType;

	SizeType Empty;

	SizeType Count;

	ArrayType Indices;

	TListHandlerCursor()
	{
		Reset();
	}

	TListHandlerCursor(SizeType Size, bool bSkipInit = false)
	{
		Reset(Size, bSkipInit);
	}

	FORCEINLINE SizeType Num() const
	{
		return Count;
	}

	FORCEINLINE void Reset(bool bSkipInit = false)
	{
		Empty = 0;
		Count = 0;
		if (!bSkipInit)
		{
			for (SizeType Index = 0; Index < Indices.Num(); ++Index) { Indices[Index] = Index + 1; }
		}
	}

	FORCEINLINE void Reset(SizeType Size, bool bSkipInit = false, EAllowShrinking AllowShrinking = EAllowShrinking::Yes)
	{
		const bool bSize = Size > 0;
		if (bSize) { Indices.SetNum(Size, AllowShrinking); }
		Reset(bSkipInit);
	}

	FORCEINLINE bool IsEmpty() const
	{
		return Count <= 0;
	}
};

using FListHandlerCursor = TListHandlerCursor<FDefaultAllocator>;

template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
class VACTBASE_API TListHandler
{
	template<typename OtherInElementType, typename OtherAllocator>
	friend class TListHandler;

public:
	typedef TListHandlerCursor<typename InAllocatorType> CursorType;
	typedef typename InAllocatorType::SizeType SizeType;
	typedef InElementType ElementType;
	typedef TArray<typename InElementType, typename InAllocatorType> ArrayType;
	typedef TIterateHandler<TListHandler<typename InElementType, typename InAllocatorType>> IteratorType;

private:
	CursorType _Cursor;

	ArrayType* _ArrayPtr;

public:
	TListHandler() : TListHandler(nullptr)
	{
	}

	TListHandler(ArrayType* ArrayPtr, bool bSkipInit = false) : _ArrayPtr(ArrayPtr)
	{
		//_Cursor = CursorType(Max(), bSkipInit);
		_Cursor.Reset(ArrayPtr ? Max() : 0, bSkipInit);
	}

	TListHandler(ArrayType& Array, bool bSkipInit = false) : TListHandler(&Array)
	{

	}

	~TListHandler()
	{
		_ArrayPtr = nullptr;
	}

	/*FORCEINLINE*/ SizeType Add(ElementType& Element)
	{
		SizeType Index = _Cursor.Empty;
		_Cursor.Empty = _Cursor.Indices[Index];
		_Cursor.Indices[Index] = -1;
		(*_ArrayPtr)[Index] = Element;
		++_Cursor.Count;
		return Index;
	}

	FORCEINLINE SizeType Add(ElementType& Element, SizeType& Index)
	{
		Index = Add(Element);
		return Index;
	}

	/*FORCEINLINE*/ ElementType& Remove(SizeType Index)
	{
		--_Cursor.Count;
		_Cursor.Indices[Index] = _Cursor.Empty;
		_Cursor.Empty = Index;
		return (*_ArrayPtr)[Index];
	}

	FORCEINLINE ElementType& Remove(SizeType Index, ElementType& Into)
	{
		Into = (*_ArrayPtr)[Index];
		return Remove(Index);
	}

	FORCEINLINE void RemoveNull(SizeType Index, ElementType NullDefault)
	{
		Remove(Index);
		(*_ArrayPtr)[Index] = NullDefault;
	}

	FORCEINLINE void RemoveNull(SizeType Index, ElementType& Into, ElementType NullDefault)
	{
		Into = (*_ArrayPtr)[Index];
		RemoveNull(Index, NullDefault);
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

	FORCEINLINE void Empty(bool bSkipInit = false)
	{
		_Cursor.Reset(bSkipInit);
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

	bool _Next(SizeType& Counter, SizeType& Tracker, SizeType& Index) const
	{
		Index = -1;
		ArrayType& _Array = (*_ArrayPtr);

		bool bNext = _Cursor.Count > 0 /*&& Counter < _Cursor.Count*/ && Tracker >= 0 && Tracker < _Array.Num();
		bool bSkip = bNext && _Cursor.Indices[Tracker] >= 0;
		
		while (bSkip)
		{
			++Tracker;
			bNext = Tracker < _Array.Num();
			bSkip = bNext && _Cursor.Indices[Tracker] >= 0;
		}

		if (bNext)
		{
			Index = Tracker; 
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