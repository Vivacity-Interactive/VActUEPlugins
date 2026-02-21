#pragma once

#include "CoreMinimal.h"
#include "VActMathTypes.generated.h"

#define _VACTMATH_OUTOFBOUNDS_MSG TEXT("index out of bounds: %lld into an array of size %lld")
#define _VACTMATH_MISMATCH_MSG TEXT("size mismatch: expected %lld, got %lld")

UENUM()
enum class EMathOperation
{
	None,
	Add,
	Sub,
	Mul,
	Div,
	Pow,
	Log,
	Mod,
	Max,
	Min,
	Mean,
	Rms,
	Mse,
	Skew,
	Or,
	And,
	Xor,
	Diff,
	Eq,
	Neq,
	Lt,
	Gt,
	Leq,
	Geq,
	Assign,
	Move,
	Swap,
	Snap,
	ExpSnap,
	Round,
	Sign,
	Abs
};


struct FVActMathConst
{
	FVActMathConst() = delete;

	template<typename T0>
	static constexpr T0 _Smt_A2{ static_cast<T0>(0.48) };

	template<typename T0>
	static constexpr T0 _Smt_A3{ static_cast<T0>(0.235f) };

	template<typename T0>
	static constexpr T0 Zero{ static_cast<T0>(0) };

	template<typename T0>
	static constexpr T0 Third{ static_cast<T0>(1 / 3) };

	template<typename T0>
	static constexpr T0 NegThird{ static_cast<T0>(-1 / 3) };

	template<typename T0>
	static constexpr T0 Half{ static_cast<T0>(0.5) };

	template<typename T0>
	static constexpr T0 NegHalf{ static_cast<T0>(0.5) };

	template<typename T0>
	static constexpr T0 One{ static_cast<T0>(1) };

	template<typename T0>
	static constexpr T0 NegOne{ static_cast<T0>(-1) };

	template<typename T0>
	static constexpr T0 Two{ static_cast<T0>(2) };

	template<typename T0>
	static constexpr T0 NegTwo{ static_cast<T0>(-2) };

	template<typename T0>
	static constexpr T0 Three{ static_cast<T0>(3) };

	template<typename T0>
	static constexpr T0 NegThree{ static_cast<T0>(-3) };

	template<typename T0>
	static constexpr T0 Eps{ TNumericLimits<T0>::Epsilon() };

	template<typename T0>
	static constexpr T0 Small{ static_cast<T0>(KINDA_SMALL_NUMBER) };

	template<typename T0>
	static constexpr T0 True{ static_cast<T0>(true) };

	template<typename T0>
	static constexpr T0 False{ static_cast<T0>(false) };

	template<typename T0>
	static constexpr T0 Max{ TNumericLimits<T0>::Max() };

	template<typename T0>
	static constexpr T0 Min{ TNumericLimits<T0>::Min() };
};

template<typename T0, uint32 N0>
struct TMathVector
{
	T0 Data[N0];

	constexpr FORCEINLINE int32 Num() const
	{
		return static_cast<int32>(N0);
	}

	constexpr FORCEINLINE SIZE_T NumBytes() const
	{
		return sizeof(Data);
	}

	FORCEINLINE bool IsValidIndex(int32 Index) const
	{
		return Index >= 0 && Index < static_cast<int32>(N0);
	}

	FORCEINLINE void RangeCheck(int32 Index) const
	{
		checkf((Index >= 0) & (Index < static_cast<int32>(N0)), _VACTMATH_OUTOFBOUNDS_MSG, (long long)Index, (long long)N0);
	}

	FORCEINLINE operator T0* ()
	{
		return Data;
	}

	FORCEINLINE operator const T0* () const
	{
		return Data;
	}

	FORCEINLINE T0& operator[](int32 Index)
	{
		RangeCheck(Index);
		return Data[Index];
	}

	FORCEINLINE const T0& operator[](int32 Index) const
	{
		RangeCheck(Index);
		return Data[Index];
	}
};

template<uint32 N0>
using TFloatMathVector = TMathVector<float, N0>;

template<uint32 N0>
using TMathEffector = TMathVector<EMathOperation, N0>;

template<typename T0, typename TypeMathVector>
using TMathVectorOf = TMathVector<T0, sizeof(TypeMathVector) / sizeof(T0)>;

template<typename T0, typename TypeMathVector>
using TMathEffectorOf = TMathVector<EMathOperation, sizeof(TypeMathVector) / sizeof(T0)>;

template<typename T0>
struct TWeightedItem
{
	T0 Item;
	
	float Weight;

	TWeightedItem()
		: Weight(1.0f)
	{
	}
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTMATH_API FWeightedIndex
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Index;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Weight;

	FWeightedIndex();
};

template<typename T0>
struct TVectorBuffer
{
	T0* Data;
	
	SIZE_T Size;
	
	int32 Count;
	
	int32 VectorSize;

	FORCEINLINE int32 Num() const
	{
		return Count;
	}

	FORCEINLINE T0* operator*()
	{
		return Data;
	}

	const FORCEINLINE T0* operator*() const
	{
		return Data;
	}

	template<typename T1>
	T1& operator[](int32 Index)
	{
		checkf(sizeof(T1) / sizeof(T0) == VectorSize, _VACTMATH_MISMATCH_MSG, (long long)(VectorSize), (long long)(sizeof(T1)/sizeof(T0)));
		checkf(Index >= 0 && Index < Count, _VACTMATH_OUTOFBOUNDS_MSG, (long long)Index, (long long)Count);
		return *reinterpret_cast<T1*>(Data + Index * VectorSize);
	}

	template<typename T1>
	const T1& operator[](int32 Index) const
	{
		checkf(sizeof(T1) / sizeof(T0) == VectorSize, _VACTMATH_MISMATCH_MSG, (long long)(VectorSize), (long long)(sizeof(T1) / sizeof(T0)));
		checkf(Index >= 0 && Index < Count, _VACTMATH_OUTOFBOUNDS_MSG, (long long)Index, (long long)Count);
		return *reinterpret_cast<T1*>(Data + Index * VectorSize);
	}

	template<typename T1>
	T1* begin()
	{
		checkf(sizeof(T1) / sizeof(T0) == VectorSize, _VACTMATH_MISMATCH_MSG, (long long)(VectorSize), (long long)(sizeof(T1) / sizeof(T0)));
		return Data;
	}

	template<typename T1>
	const T1* begin() const
	{
		checkf(sizeof(T1) / sizeof(T0) == VectorSize, _VACTMATH_MISMATCH_MSG, (long long)(VectorSize), (long long)(sizeof(T1) / sizeof(T0)));
		return Data;
	}

	template<typename T1>
	T1* end()
	{
		checkf(sizeof(T1) / sizeof(T0) == VectorSize, _VACTMATH_MISMATCH_MSG, (long long)(VectorSize), (long long)(sizeof(T1) / sizeof(T0)));
		return Data + Count;
	}

	template<typename T1>
	const T1* end() const
	{
		checkf(sizeof(T1) / sizeof(T0) == VectorSize, _VACTMATH_MISMATCH_MSG, (long long)(VectorSize), (long long)(sizeof(T1) / sizeof(T0)));
		return Data + Count;
	}
};

template<typename T0>
struct TStructBuffer
{
	T0* Data;
	
	SIZE_T Size;
	
	int32 Count;

	FORCEINLINE int32 Num() const
	{
		return Count;
	}

	FORCEINLINE T0* operator*()
	{
		return Data;
	}

	const FORCEINLINE T0* operator*() const
	{
		return Data;
	}

	T0& operator[](int32 Index)
	{
		checkf(Index >= 0 && Index < Count, _VACTMATH_OUTOFBOUNDS_MSG, (long long)Index, (long long)Count);
		return Data[Index];
	}

	const T0& operator[](int32 Index) const
	{
		checkf(Index >= 0 && Index < Count, _VACTMATH_OUTOFBOUNDS_MSG, (long long)Index, (long long)Count);
		return Data[Index];
	}

	T0* begin()
	{
		return Data;
	}

	const T0* begin() const
	{
		return Data;
	}

	T0* end()
	{
		return Data + Count;
	}

	const T0* end() const
	{
		return Data + Count;
	}
};


