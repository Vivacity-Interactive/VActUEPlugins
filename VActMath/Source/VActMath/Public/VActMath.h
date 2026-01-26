#pragma once

#include "VActMathTypes.h"
#include "CoreMinimal.h"

struct VACTMATH_API FVActMath
{
	FVActMath() = delete;


	// Base Functions

	template<typename T0 = float>
	FORCEINLINE static T0 Pry(const T0 X, const T0 Min, const T0 Max)
	{
		const T0 Pivot = Min + (Max - Min) * FVActMathConst::Half;
		return X <= Min ? X : (X >= Max ? X : (X < Pivot ? Min : Max));
	}

	template<typename T0 = float>
	FORCEINLINE static T0 Round(const T0 X)
	{
		return TIsFloatingPoint<T0>::Value ? ((X >= FVActMathConst::Zero) ? X + FVActMathConst::Half : X - FVActMathConst::Half) : X;
	}

	template<typename T0 = float>
	FORCEINLINE static T0 Snap(const T0 X, const T0 Step)
	{
		return RoundTo(X / Step) * Step;
	}

	template<typename T0 = float>
	FORCEINLINE static T0 ExpSnap(const T0 X, const T0 Step, const T0 Exp)
	{
		return RoundTo(FMath::Pow(X, Exp) / Step) * Step;
	}

	template<typename T0 = float>
	FORCEINLINE static T0 Mix(const T0 A, const T0 B, const T0 Alpha, const T0 Beta, const T0 Eps = FVActMathConst::Small)
	{
		const T0 Norm = Alpha + Beta;
		return Norm <= Eps ? 0.0f : (A * Alpha + B * Beta) / Norm;
	}

	template<typename T0 = float>
	FORCEINLINE static T0 SmoothTo(T0 Current, T0 Target, T0& Velocity, T0 Delta, T0 Alpha, T0 Limit, const T0 Eps = FVActMathConst::Small)
	{
		T0 Into;

		T0 _Alpha = FMath::Max(Eps, Alpha);
		T0 Omega = FVActMathConst::Two / _Alpha;
		T0 X = Omega * Delta;
		T0 Exp = FVActMathConst::One / (FVActMathConst::One + X + FVActMathConst::_Smt_A2 * X * X + FVActMathConst::_Smt_A3 * X * X * X);
		T0 Gamma = FVActMathConst::One / Delta;

		T0 _Change = Current - Target;
		T0 _Target = Target;

		T0 MaxChange = Limit * _Alpha;
		T0 MaxChangeSq = MaxChange * MaxChange;

		T0 SqrLength = _Change * _Change;
		if (SqrLength > MaxChangeSq)
		{
			T0 Length = FMath::Sqrt(SqrLength);
			T0 Beta = FVActMathConst::One / (Length * MaxChange);
			_Change *= Beta;
		}

		T0 NewTarget = Current - _Change;
		//float NewTarget = Current + _Change;
		T0 Temp = (Velocity + Omega * _Change) * Delta;

		Velocity = (Velocity - Omega * Temp) * Exp;
		Into = NewTarget + (_Change + Temp) * Exp;

		T0 DiffTarget = _Target - Current;
		T0 DiffCurrent = Into - _Target;

		T0 Product = DiffTarget * DiffCurrent;

		//float Sum = Product.X + Product.Y + Product.Z;
		if (Product > Eps)
		{
			Into = _Target;
			//Velocity = (Into - _Target) * Gamma;
			Velocity = FVActMathConst::Zero;
		}

		return Into;
	}


	// General Enum Functions

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Round(T0* Into, const T0* A, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = RoundTo(A[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_ExpSnap(T0* Into, const T0* X, const T0* Step, const T0* Exp, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = ExpSnap(X[Index], Step[Index], Exp[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_ExpSnap(T0* Into, const T0* X, const T0 Step, const T0* Exp, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = ExpSnap(X[Index], Step, Exp[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_ExpSnap(T0* Into, const T0* X, const T0* Step, const T0 Exp, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = ExpSnap(X[Index], Step[Index], Exp);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_ExpSnap(T0* Into, const T0* X, const T0 Step, const T0 Exp, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = ExpSnap(X[Index], Step, Exp);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Snap(T0* Into, const T0* X, const T0* Step, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Snap(X[Index], Step[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Snap(T0* Into, const T0* X, const T0 Step, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Snap(X[Index], Step);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Snap(T0* Into, const T0 X, const T0* Step, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Snap(X, Step[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Assign(T0* Into, const T0* A, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Assign(T0* Into, const T0 A, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A;
		}
	}
	
	template<typename T0=float>
	FORCEINLINE static void _Unsafe_Add(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			Into[Index] = A[Index] + B[Index];
		}
	}

	template<typename T0=float>
	FORCEINLINE static void _Unsafe_Add(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			Into[Index] = A[Index] + B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mul(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] * B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mul(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] * B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Sub(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] - B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Sub(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] - B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Sub(T0* Into, const T0 A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A - B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Div(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] / B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Div(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] / B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Div(T0* Into, const T0 A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A / B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Swap(T0* A, T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 _A = A[Index];
			A[Index] = B[Index];
			B[Index] = _A;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Move(T0* A, T0* B, const int32 Count, const T0 Null = FVActMathConst::Zero)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			A[Index] = B[Index];
			B[Index] = Null;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Move(T0* A, T0* B, const int32 Count, const T0* Null)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			A[Index] = B[Index];
			B[Index] = Null[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pow(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Pow(A[Index], B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pow(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Pow(A[Index], B);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pow(T0* Into, const T0 A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Pow(A, B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Log(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::LogX(A[Index], B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Log(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::LogX(A[Index], B);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Log(T0* Into, const T0 A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::LogX(A, B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mod(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Fmod(A[Index], B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mod(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Fmod(A[Index], B);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mod(T0* Into, const T0 A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Fmod(A, B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Max(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Max(A[Index], B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Max(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Max(A[Index], B);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Min(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Min(A[Index], B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Min(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Min(A[Index], B);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mean(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = (A[Index] + B[Index]) * FVActMathConst::Half;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mean(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = (A[Index] + B) * FVActMathConst::Half;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Rms(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Sqrt((A[Index] * A[Index] + B[Index] * B[Index]) * FVActMathConst::Half);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Rms(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Sqrt((A[Index] * A[Index] + B * B) * FVActMathConst::Half);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mse(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 Delta = B[Index] - A[Index];
			Into[Index] = Delta * Delta * FVActMathConst::Half;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Skew(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = (A[Index] - B[Index]) / (FMath::Abs(A[Index]) + FMath::Abs(B[Index]) + Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Skew(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = (A[Index] - B) / (FMath::Abs(A[Index]) + FMath::Abs(B) + Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Or(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] | B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Or(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] | B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_And(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] & B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_And(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] & B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Xor(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] ^ B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Xor(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] ^ B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Diff(T0* Into, const T0* A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] & ~B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Diff(T0* Into, const T0* A, const T0 B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A[Index] & ~B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Diff(T0* Into, const T0 A, const T0* B, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = A & ~B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Eq(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(FMath::IsNearlyEqual(A[Index], B[Index], Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Eq(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(FMath::IsNearlyEqual(A[Index], B, Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Neq(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(!FMath::IsNearlyEqual(A[Index], B[Index], Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Neq(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(!FMath::IsNearlyEqual(A[Index], B, Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lt(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] < (B[Index] - Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lt(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] < (B - Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Gt(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] > (B[Index] + Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Gt(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] > (B + Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Leq(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] <= (B[Index] + Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Leq(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] <= (B + Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Geq(T0* Into, const T0* A, const T0* B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] >= (B[Index] - Eps));
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Geq(T0* Into, const T0* A, const T0 B, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = static_cast<T0>(A[Index] >= (B - Eps));
		}
	}


	// General Enum Apply Function

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Apply(T0* A, T0* B, EMathOperation Opp, const int32 Count, const T0 Null = FVActMathConst::Zero, const T0 Eps = FVActMathConst::Eps)
	{
		switch (Opp)
		{
		case EMathOperation::Add:
			_Unsafe_Add(A, A, B, Count);
			break;
		case EMathOperation::Sub:
			_Unsafe_Sub(A, A, B, Count);
			break;
		case EMathOperation::Mul:
			_Unsafe_Mul(A, A, B, Count);
			break;
		case EMathOperation::Div:
			_Unsafe_Div(A, A, B, Count);
			break;
		case EMathOperation::Pow:
			_Unsafe_Pow(A, A, B, Count);
			break;
		case EMathOperation::Log:
			_Unsafe_Log(A, A, B, Count);
			break;
		case EMathOperation::Mod:
			_Unsafe_Mod(A, A, B, Count);
			break;
		case EMathOperation::Max:
			_Unsafe_Max(A, A, B, Count);
			break;
		case EMathOperation::Min:
			_Unsafe_Min(A, A, B, Count);
			break;
		case EMathOperation::Mean:
			_Unsafe_Mean(A, A, B, Count);
			break;
		case EMathOperation::Rms:
			_Unsafe_Rms(A, A, B, Count);
			break;
		case EMathOperation::Mse:
			_Unsafe_Mse(A, A, B, Count);
			break;
		case EMathOperation::Skew:
			_Unsafe_Skew(A, A, B, Count, Eps);
			break;
		case EMathOperation::Or:
			_Unsafe_Or(A, A, B, Count);
			break;
		case EMathOperation::And:
			_Unsafe_And(A, A, B, Count);
			break;
		case EMathOperation::Xor:
			_Unsafe_Xor(A, A, B, Count);
			break;
		case EMathOperation::Diff:
			_Unsafe_Diff(A, A, B, Count);
			break;
		case EMathOperation::Eq:
			_Unsafe_Eq(A, A, B, Count, Eps);
			break;
		case EMathOperation::Neq:
			_Unsafe_Neq(A, A, B, Count, Eps);
			break;
		case EMathOperation::Lt:
			_Unsafe_Lt(A, A, B, Count, Eps);
			break;
		case EMathOperation::Gt:
			_Unsafe_Gt(A, A, B, Count, Eps);
			break;
		case EMathOperation::Leq:
			_Unsafe_Leq(A, A, B, Count, Eps);
			break;
		case EMathOperation::Geq:
			_Unsafe_Geq(A, A, B, Count, Eps);
			break;
		case EMathOperation::Assign:
			_Unsafe_Assign(A, B, Count);
			break;
		case EMathOperation::Move:
			_Unsafe_Move(A, A, Count, Null);
			break;
		case EMathOperation::Swap:
			_Unsafe_Swap(A, B, Count);
			break;
		case EMathOperation::Snap:
			_Unsafe_Snap(A, B, Count);
			break;
		case EMathOperation::ExpSnap:
			_Unsafe_Exp(A, A, B, Count);
			break;
		case EMathOperation::Round:
			_Unsafe_Round(A, B, Count);
			break;
		default:
			break;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Apply(T0* A, T0* B, EMathOperation* Opp, const int32 Count, const T0 Null = FVActMathConst::Zero, const T0 Eps = FVActMathConst::Eps)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			switch (Opp[Index])
			{
			case EMathOperation::Add:
				A[Index] = A[Index] + B[Index];
				break;
			case EMathOperation::Sub:
				A[Index] = A[Index] - B[Index];
				break;
			case EMathOperation::Mul:
				A[Index] = A[Index] * B[Index];
				break;
			case EMathOperation::Div:
				A[Index] = A[Index] / B[Index];
				break;
			case EMathOperation::Pow:
				A[Index] = FMath::Pow(A[Index], B[Index]);
				break;
			case EMathOperation::Log:
				A[Index] = FMath::LogX(A[Index], B[Index]);
				break;
			case EMathOperation::Mod:
				A[Index] = FMath::Fmod(A[Index], B[Index]);
				break;
			case EMathOperation::Max:
				A[Index] = FMath::Max(A[Index], B[Index]);
				break;
			case EMathOperation::Min:
				A[Index] = FMath::Min(A[Index], B[Index]);
				break;
			case EMathOperation::Mean:
				A[Index] = (A[Index] + B[Index]) * FVActMathConst::Half;
				break;
			case EMathOperation::Rms:
				A[Index] = FMath::Sqrt((A[Index] * A[Index] + B[Index] * B[Index]) * FVActMathConst::Half);
				break;
			case EMathOperation::Mse:
				const T0 Delta = B[Index] - A[Index];
				A[Index] = Delta * Delta * FVActMathConst::Half;
				break;
			case EMathOperation::Skew:
				A[Index] = (A[Index] - B[Index]) / (FMath::Abs(A[Index]) + FMath::Abs(B[Index]) + Eps);
				break;
			case EMathOperation::Or:
				A[Index] = A[Index] | B[Index];
				break;
			case EMathOperation::And:
				A[Index] = A[Index] & B[Index];
				break;
			case EMathOperation::Xor:
				A[Index] = A[Index] ^ B[Index];
				break;
			case EMathOperation::Diff:
				A[Index] = A[Index] & ~B[Index];
				break;
			case EMathOperation::Eq:
				A[Index] = static_cast<T0>(FMath::IsNearlyEqual(A[Index], B[Index], Eps));
				break;
			case EMathOperation::Neq:
				A[Index] = static_cast<T0>(!FMath::IsNearlyEqual(A[Index], B[Index], Eps));
				break;
			case EMathOperation::Lt:
				A[Index] = static_cast<T0>(A[Index] < (B[Index] - Eps));
				break;
			case EMathOperation::Gt:
				A[Index] = static_cast<T0>(A[Index] > (B[Index] + Eps));
				break;
			case EMathOperation::Leq:
				A[Index] = static_cast<T0>(A[Index] <= (B[Index] + Eps));
				break;
			case EMathOperation::Geq:
				A[Index] = static_cast<T0>(A[Index] >= (B[Index] - Eps));
				break;
			case EMathOperation::Assign:
				A[Index] = B[Index];
				break;
			case EMathOperation::Move:
				A[Index] = B[Index];
				B[Index] = Null;
				break;
			case EMathOperation::Swap:
				const T0 _A = A[Index];
				A[Index] = B[Index];
				B[Index] = _A;
				break;
			case EMathOperation::Snap:
				A[Index] = Snap(A[Index], B[Index]);
				break;
			case EMathOperation::ExpSnap:
				A[Index] = ExpSnap(A[Index], A[Index], B[Index]);
				break;
			case EMathOperation::Round:
				A[Index] = Round(B[Index]);
				break;
			default:
				break;
			}
		}
	}


	// Other Math Functions

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Kernel(T0* Into, const T0* X, const T0* Kernel, const int32 Count, const int32 Window, const int32 Pivot = 0)
	{
		const int32 _Window = FMath::Min(Window, Count);

		for (int32 Index = 0; Index < Count; ++Index)
		{
			T0 Sum = FVActMathConst::Zero;
			for (int32 Jndex = 0; Jndex < _Window; ++Jndex)
			{
				const int32 _Index = FMath::Clamp(Index + Jndex - Pivot, 0, Count - 1);
				Sum += X[_Index] * Kernel[Jndex] * static_cast<T0>(Jndex - Pivot >= 0);
			}
			Into[Index] = Sum;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_RowApply(T0* Into, const T0* X, const T0* Matrix, const int32 Count)
	{
		for (int32 Row = 0; Row < Count; ++Row)
		{
			T0 Sum = FVActMathConst::Zero;
			for (int32 Col = 0; Col < Count; ++Col)
			{
				const int32 _Index = Row * Count + Col;
				Sum += X[Col] * Matrix[_Index];
			}
			Into[Row] = Sum;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_ColApply(T0* Into, const T0* X, const T0* Matrix, const int32 Count)
	{
		for (int32 Row = 0; Row < Count; ++Row)
		{
			T0 Sum = FVActMathConst::Zero;
			for (int32 Col = 0; Col < Count; ++Col)
			{
				const int32 _Index = Row + Col * Count;
				Sum += X[Col] * Matrix[_Index];
			}
			Into[Row] = Sum;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_RowApply(T0* Into, const T0* X, const T0* Matrix, const int32 Rows, const int32 Cols)
	{
		for (int32 Row = 0; Row < Rows; ++Row)
		{
			T0 Sum = FVActMathConst::Zero;
			for (int32 Col = 0; Col < Cols; ++Col)
			{
				const int32 _Index = Row * Cols + Col;
				Sum += X[Col] * Matrix[_Index];
			}
			Into[Row] = Sum;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_ColApply(T0* Into, const T0* X, const T0* Matrix, const int32 Rows, const int32 Cols)
	{
		for (int32 Row = 0; Row < Rows; ++Row)
		{
			T0 Sum = FVActMathConst::Zero;
			for (int32 Col = 0; Col < Cols; ++Col)
			{
				const int32 _Index = Row + Col * Rows;
				Sum += X[Col] * Matrix[_Index];
			}
			Into[Row] = Sum;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Clamp(T0* Into, const T0* X, const T0* Min, const T0* Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Clamp(X[Index], Min[Index], Max[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Clamp(T0* Into, const T0* X, const T0* Min, const T0 Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Clamp(X[Index], Min[Index], Max);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Clamp(T0* Into, const T0* X, const T0 Min, const T0* Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Clamp(X[Index], Min, Max[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Clamp(T0* Into, const T0* X, const T0 Min, const T0 Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Clamp(X[Index], Min, Max);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pry(T0* Into, const T0* X, const T0* Min, const T0* Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Pry(X[Index], Min[Index], Max[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pry(T0* Into, const T0* X, const T0* Min, const T0 Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Pry(X[Index], Min[Index], Max);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pry(T0* Into, const T0* X, const T0 Min, const T0* Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Pry(X[Index], Min, Max[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Pry(T0* Into, const T0* X, const T0 Min, const T0 Max, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Pry(X[Index], Min, Max);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0* B, const T0* Alpha, const T0* Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B[Index], Alpha[Index], Beta[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0* B, const T0* Alpha, const T0 Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B[Index], Alpha, Beta[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0* B, const T0 Alpha, const T0* Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B[Index], Alpha[Index], Beta);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0 B, const T0* Alpha, const T0* Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B, Alpha[Index], Beta[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0* B, const T0 Alpha, const T0 Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B[Index], Alpha, Beta);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0 B, const T0* Alpha, const T0 Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B, Alpha, Beta[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0 B, const T0 Alpha, const T0* Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B, Alpha[Index], Beta);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mix(T0* Into, const T0* A, const T0 B, const T0 Alpha, const T0 Beta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = Mix(A[Index], B, Alpha, Beta);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lerp(T0* Into, const T0* A, const T0* B, const T0* Alpha, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Lerp(A[Index], B[Index], Alpha[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lerp(T0* Into, const T0* A, const T0* B, const T0 Alpha, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Lerp(A[Index], B[Index], Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lerp(T0* Into, const T0* A, const T0 B, const T0 Alpha, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Lerp(A[Index], B, Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lerp(T0* Into, const T0 A, const T0* B, const T0* Alpha, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Lerp(A, B[Index], Alpha[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Lerp(T0* Into, const T0 A, const T0* B, const T0 Alpha, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::Lerp(A, B[Index], Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0* Target, const T0* Alpha, const T0* Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target[Index], Delta[Index], Alpha[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0* Target, const T0* Alpha, const T0 Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target[Index], Delta, Alpha[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0* Target, const T0 Alpha, const T0* Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target[Index], Delta[Index], Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0 Target, const T0* Alpha, const T0* Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target, Delta[Index], Alpha[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0* Target, const T0 Alpha, const T0 Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target[Index], Delta, Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0 Target, const T0* Alpha, const T0 Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target, Delta, Alpha[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0 Target, const T0 Alpha, const T0* Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target, Delta[Index], Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_InterpTo(T0* Into, const T0* Current, const T0 Target, const T0 Alpha, const T0 Delta, const int32 Count)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = FMath::FInterpTo(Current[Index], Target, Delta, Alpha);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0* Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta[Index], Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0* Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta, Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0 Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta[Index], Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0* Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta[Index], Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0 Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta, Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0* Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta, Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0 Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta[Index], Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0 Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta, Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0* Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta[Index], Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0* Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta, Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0 Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta[Index], Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0* Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta[Index], Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0 Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta, Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0* Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta, Alpha[Index], Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0 Alpha, const T0* Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta[Index], Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0 Alpha, const T0 Delta, const T0* Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta, Alpha, Limit[Index], Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0* Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta[Index], Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0* Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta, Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0 Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta[Index], Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0* Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta[Index], Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0* Velocity, const T0 Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity[Index], Delta, Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0* Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta, Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0 Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta[Index], Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0* Velocity, const T0 Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity[Index], Delta, Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0* Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta[Index], Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0* Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta, Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0 Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta[Index], Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0* Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta[Index], Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0* Target, const T0 Velocity, const T0 Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target[Index], Velocity, Delta, Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0* Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta, Alpha[Index], Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0 Alpha, const T0* Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta[Index], Alpha, Limit, Eps);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SmoothTo(T0* Into, const T0* Current, const T0 Target, const T0 Velocity, const T0 Alpha, const T0 Delta, const T0 Limit, const int32 Count, const T0 Eps = FVActMathConst::Small)
	{
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] = SmoothTo(Current[Index], Target, Velocity, Delta, Alpha, Limit, Eps);
		}
	}


	// Analitic Functions

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Dot(T0& Into, const T0* A, const T0* B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] += A[Index] * B[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Dot(T0& Into, const T0* A, const T0 B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] += A[Index] * B;
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_AbsDot(T0& Into, const T0* A, const T0* B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] += FMath::Abs(A[Index] * B[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_AbsDot(T0& Into, const T0* A, const T0 B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into[Index] += FMath::Abs(A[Index] * B);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Sum(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into += A[Index];
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_AbsSum(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into += FMath::Abs(A[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Length(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into += A[Index] * A[Index];
		}
		Into = FMath::Sqrt(Into);
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Dist(T0& Into, const T0* A, const T0* B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 Delta = B[Index] - A[Index];
			Into += Delta * Delta;
		}
		Into = FMath::Sqrt(Into);
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Dist(T0& Into, const T0* A, const T0 B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 Delta = B - A[Index];
			Into += Delta * Delta;
		}
		Into = FMath::Sqrt(Into);
	}


	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SignDist(T0& Into, const T0* A, const T0* B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		const T0 _Sign = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 Delta = B[Index] - A[Index];
			_Sign += A[Index] * B[Index];
			Into += Delta * Delta;
		}
		Into = FMath::Sqrt(Into) * FMath::Sign(_Sign);
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_SignDist(T0& Into, const T0* A, const T0 B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		T0 _Sign = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 Delta = B - A[Index];
			_Sign += A[Index] * B;
			Into += Delta * Delta;
		}
		Into = FMath::Sqrt(Into) * FMath::Sign(_Sign);
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Max(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Min;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into = FMath::Max(Into, A[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Min(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Max;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into = FMath::Min(Into, A[Index]);
		}
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mean(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into += A[Index];
		}
		Into /= static_cast<T0>(Count);
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Rms(T0& Into, const T0* A, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Into += A[Index] * A[Index];
		}
		Into = FMath::Sqrt(Into / static_cast<T0>(Count));
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Mse(T0& Into, const T0* A, const T0* B, const int32 Count)
	{
		Into = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			const T0 Delta = A[Index] - B[Index];
			Into += Delta * Delta;
		}
		Into /= static_cast<T0>(Count);
	}

	template<typename T0 = float>
	FORCEINLINE static void _Unsafe_Skew(T0& Into, const T0* A, const int32 Count, const T0 Eps = FVActMathConst::Eps)
	{
		T0 Mean;
		_Unsafe_Mean(Mean, A, Count);
		Into = FVActMathConst::Zero;
		const T0 Norm = FVActMathConst::Zero;
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Norm += FMath::Abs(A[Index]);
			Into += A[Index] - Mean;
		}
		Into = Into / (Norm + Eps);
	}

};
