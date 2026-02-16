#pragma once

#include "CoreMinimal.h"
#include "VActMathTypes.generated.h"

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

	FORCEINLINE int32 Num() const
	{
		return N0;
	}

	FORCEINLINE SIZE_T NumBytes() const
	{
		return sizeof(Data);
	}

	FORCEINLINE bool IsValidIndex(int32 Index) const
	{
		return Index >= 0 && Index < static_cast<int32>(N0);
	}

	FORCEINLINE void RangeCheck(int32 Index) const
	{
		checkf((Index >= 0) & (Index < static_cast<int32>(N0)), TEXT("MathVector index out of bounds: %lld into an array of size %lld"), (long long)Index, (long long)NumFeatures);
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
using TMathEffector = TMathVector<EMathOperation, N0>;

template<typename T0, uint32 N0>
struct TMathMonoEffect
{
	EMathOperation Effector;
	TMathVector<T0, N0> Vector;
};

template<typename T0, uint32 N0>
struct TMathEffect
{
	TMathEffector<N0> Effector;
	TMathVector<T0, N0> Vector;
};

template<typename T0>
struct TWeightedEntry
{
	T0 Index;
	float Weight;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTMATH_API FWeightedEntry
{
	GENERATED_BODY()

	int32 Index;
	float Weight;

	FWeightedEntry();
};


template<typename T0, typename TypeMathVector>
using TMathVectorOf = TMathVector<sizeof(TypeMathVector) / sizeof(T0)>;

template<typename T0, typename TypeMathVector>
using TMathEffectorOf = TMathEffector<sizeof(TypeMathVector) / sizeof(T0)>;

template<typename T0, typename TypeMathVector>
using TMathEffectOf = TMathEffect<T0, sizeof(TypeMathVector) / sizeof(T0)>;

template<typename T0, typename TypeMathVector>
using TMathMonoEffectOf = TMathMonoEffect<T0, sizeof(TypeMathVector) / sizeof(T0)>;

template<uint32 N0>
using TFloatMathVector = TMathVector<float, N0>;

template<uint32 N0>
using TFloatMathEffect = TMathEffect<float, N0>;

template<uint32 N0>
using TFloatMathMonoEffect = TMathMonoEffect<float, N0>;