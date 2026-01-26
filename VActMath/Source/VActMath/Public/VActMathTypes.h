#pragma once

#include "CoreMinimal.h"

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
	Round
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