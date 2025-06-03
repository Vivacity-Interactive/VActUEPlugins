#pragma once

#include "CoreMinimal.h"
#include "Containers/ContainerAllocationPolicies.h"
#include "Math/UnrealMathUtility.h"

template<typename InAllocatorType = FDefaultAllocator>
struct VACTBASE_API TSamplerUniqueCursor
{
private:
	template<typename OtherInAllocatorType>
	friend struct TSamplerUniqueCursor;

public:
	typedef typename InAllocatorType::SizeType SizeType;
	typedef TArray<SizeType, typename InAllocatorType> ArrayType;

	SizeType Pivot;
	ArrayType Indices;

	TSamplerUniqueCursor()
	{
		Reset();
	}
	
	TSamplerUniqueCursor(SizeType Count)
	{
		Reset(Count);
	}

	FORCEINLINE SizeType Used(SizeType Index)
	{
		Indices.Swap(Pivot, Index);
		++Pivot;

		return Index;
	}

	FORCEINLINE void Reset()
	{
		Pivot = 0;
	}

	void Reset(SizeType Count, bool bAllowShrinking = true)
	{
		SizeType _Count = Indices.Num();
		Indices.SetNum(Count, bAllowShrinking);

		for (SizeType Index = 0; Index < Indices.Num(); ++Index) { Indices[Index] = Index; }

		const bool bShrink = _Count > Indices.Num();
		if (bShrink) { Reset(); }
	}

	void SetNum(SizeType Count, bool bAllowShrinking = true)
	{
		SizeType _Count = Indices.Num();
		Indices.SetNum(Count, bAllowShrinking);
		
		SizeType _Pivot = FMath::Min(Pivot, _Count - 1);
		Pivot = 0;
		
		const bool bShrink = _Count > Indices.Num();
		SizeType Index = _Count * (_Count <= Indices.Num());
		
		for (; Index < _Pivot; ++Index)
		{
			SizeType _Index = Indices[Index];
			if (_Index < Count) { Used(Index); }
		}

		for (; Index < Indices.Num(); ++Index) { Indices[Index] = Index; }
	}

	FORCEINLINE SizeType Remaining() const
	{
		return Indices.Num() - Pivot;
	}

	FORCEINLINE bool HasRemaining() const
	{
		return Pivot >= Indices.Num();
	}
};

using FSamplerUniqueCursor = TSamplerUniqueCursor<FDefaultAllocator>;

template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
class VACTBASE_API TSampler
{
	template<typename OtherInElementType, typename OtherAllocator>
	friend class TSampler;

public:
	typedef TSamplerUniqueCursor<typename InAllocatorType> CursorType;
	typedef typename InAllocatorType::SizeType SizeType;
	typedef InElementType ElementType;
	typedef TArray<typename InElementType, typename InAllocatorType> ArrayType;
	typedef TArray<typename SizeType, typename InAllocatorType> IndicesArrayType;

private:
	ArrayType& _Samples;

	const ArrayType& _ConstSamples;

public:
	TSampler()
	{
	}

	TSampler(ArrayType& Samples) : _Samples(Samples), _ConstSamples(Samples)
	{
	}

	TSampler(const ArrayType& Samples) : _ConstSamples(Samples)
	{
	}
	
	~TSampler()
	{
	}

	FORCEINLINE ElementType& Any(ElementType& Into)
	{
		Into = Any();
		return Into;
	}

	FORCEINLINE ElementType& Any()
	{
		return _Samples[_Sample()];
	}

	FORCEINLINE const ElementType& Any() const
	{
		return _ConstSamples[_Sample()];
	}

	FORCEINLINE ElementType AnyCopy()
	{
		return _Samples[_Sample()];
	}

	FORCEINLINE SizeType AnyIndex(SizeType& Into) const
	{
		Into = _Sample();
		return Into;
	}

	ArrayType& Pick(ArrayType& Into, SizeType Count, bool bAllowShrinking = true)
	{
		Into.SetNum(Count, bAllowShrinking);
		for (SizeType Index = 0; Index < Count; ++Index) { Into[Index] = _Samples[_Sample()]; }

		return Into;
	}

	const ArrayType& Pick(SizeType Count) const
	{
		ArrayType Into;
		Into.SetNum(Count);
		for (SizeType Index = 0; Index < Count; ++Index) { Into[Index] = _ConstSamples[_Sample()]; }

		return Into;
	}

	IndicesArrayType& PickIndices(IndicesArrayType& Into, SizeType Count) const
	{
		Into.SetNum(Count);
		for (SizeType Index = 0; Index < Count; ++Index) { Into[Index] = _Sample(); }

		return Into;
	}

	FORCEINLINE ArrayType& Some(ArrayType& Into, bool bAllowShrinking = true)
	{
		return Pick(Into, _SampleCount(), bAllowShrinking);
	}

	FORCEINLINE const ArrayType& Some() const
	{
		return Pick(_SampleCount());
	}

	FORCEINLINE IndicesArrayType& SomeIncides(IndicesArrayType& Into) const
	{
		return PickIndices(Into, _SampleCount());
	}

	FORCEINLINE ElementType& AnyUnique(ElementType& Into, CursorType& Cursor)
	{
		Into = AnyUnique(Cursor);
		return Into;
	}

	FORCEINLINE ElementType& AnyUnique(CursorType& Cursor, bool bAllowShrinking = true)
	{
		return _Samples[_SampleUnique(Cursor)];
	}

	FORCEINLINE const ElementType& AnyUnique(CursorType& Cursor, bool bAllowShrinking = true) const
	{
		return _ConstSamples[_SampleUnique(Cursor)];
	}

	FORCEINLINE ElementType AnyCopyUnique(CursorType& Cursor, bool bAllowShrinking = true)
	{
		return _Samples[_SampleUnique(Cursor)];
	}

	FORCEINLINE SizeType& AnyIndexUnique(SizeType& Into, CursorType& Cursor, bool bAllowShrinking = true) const
	{
		Into = _SampleUnique(Cursor);
		return Into;
	}

	ArrayType& PickUnique(ArrayType& Into, SizeType Count, CursorType& Cursor, bool bAllowShrinking = true)
	{
		SizeType _Count = FMath::Min(Count, Cursor.Remaining());
		Into.SetNum(_Count, bAllowShrinking);
		for (SizeType Index = 0; Index < _Count; ++Index) { Into[Index] = _Samples[_SampleUnique(Cursor)]; }
		
		return Into;
	}

	const ArrayType& PickUnique(SizeType Count, CursorType& Cursor) const
	{
		const ArrayType Into;
		SizeType _Count = FMath::Min(Count, Cursor.Remaining());
		Into.SetNum(_Count);
		for (SizeType Index = 0; Index < _Count; ++Index) { Into[Index] = _ConstSamples[_SampleUnique(Cursor)]; }

		return Into;
	}

	IndicesArrayType& PickIndicesUnique(IndicesArrayType& Into, SizeType Count, CursorType& Cursor) const
	{
		SizeType _Count = FMath::Min(Count, Cursor.Remaining());
		Into.SetNum(_Count);
		for (SizeType Index = 0; Index < _Count; ++Index) { Into[Index] = _SampleUnique(Cursor); }

		return Into;
	}

	FORCEINLINE ArrayType& SomeUnique(ArrayType& Into, CursorType& Cursor, bool bAllowShrinking = true)
	{
		return PickUnique(Into, _SampleCountUnique(), Cursor, bAllowShrinking);
	}

	FORCEINLINE const ArrayType& SomeUnique(CursorType& Cursor) const
	{
		return PickUnique(_SampleCountUnique(), Cursor);
	}

	FORCEINLINE IndicesArrayType& SomeIndicesUnique(IndicesArrayType& Into, CursorType& Cursor) const
	{
		return PickIndicesUnique(Into, _SampleCountUnique(), Cursor);
	}

	FORCEINLINE ArrayType& Samples()
	{
		return _Samples;
	}

	FORCEINLINE const ArrayType& Samples() const
	{
		return _ConstSamples;
	}

	FORCEINLINE SizeType _SampleCount() const
	{
		return FMath::RandRange(0, _ConstSamples.Num() - 1);
	}

	FORCEINLINE SizeType _SampleCountUnique(CursorType& Cursor) const
	{
		return FMath::RandRange(0, Cursor.Remaining());
	}

	FORCEINLINE SizeType _Sample() const
	{
		return _SampleCount();
	}

	FORCEINLINE SizeType _SampleUnique(CursorType& Cursor) const
	{
		return Cursor.Used(FMath::RandRange(Cursor.Pivot, _ConstSamples.Num() - 1));
	}

	FORCEINLINE SizeType _SampleExclude(bool bExclude, CursorType& Cursor) const
	{
		SizeType Index = FMath::RandRange(Cursor.Pivot, _ConstSamples.Num() - 1);
		if (bExclude) { Cursor.Used(Index); }
		return Index;
	}

	FORCEINLINE void InitCursor(CursorType& Cursor, bool bAllowShrinking = true) const
	{
		Cursor.Reset(_ConstSamples.Num(), bAllowShrinking);
	}

	FORCEINLINE CursorType NewCursor() const
	{
		CursorType Cursor;
		InitCursor(Cursor);
		return Cursor;
	}
};

