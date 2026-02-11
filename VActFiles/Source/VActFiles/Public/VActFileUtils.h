#pragma once

#include "Misc/StringBuilder.h"
#include "UObject/NameTypes.h"
#include "VActFileTypes.h"

uint32 GetTypeHash(const TSharedPtr<FString>& Ptr);

struct VACTFILES_API FVActFileUtils
{
	static const TArray<TSharedPtr<FString>> FormatOptions;

	static const TMap<TSharedPtr<FString>, EVActFileFormat> MapFormat;
};

#if WITH_EDITORONLY_DATA
struct VACTFILES_API _DEBUG_VActParseInfo
{
	static const TArray<FString> TokenNames;
};
#endif

struct VACTFILES_API FVActTokenUtils
{
	static bool TokenSkipAny(const TCHAR* Chars, int32 N, FVActParseCursor& Cursor)
	{
		bool bSkip = true;

		while (bSkip && Cursor)
		{
			const TCHAR& Char = *Cursor;
			bSkip = false;
			for (int32 Index = 0; !bSkip && Index < N; ++Index)
			{
				bSkip = Chars[Index] == Char && ++Cursor;
			}
		}

		return true;
	}

	static bool TokenSkip(const TCHAR& Char, int32 N, FVActParseCursor& Cursor)
	{
		while (Char == *Cursor && ++Cursor) { }

		return true;
	}

	static bool TokenAny(const TCHAR* Chars, int32 N, FVActParseCursor& Cursor)
	{
		bool bValid = false;
		bool _bNext = Cursor.IsValid();
		const TCHAR* _To = Cursor.To;

		while (_bNext)
		{
			const TCHAR& Char = *Cursor;
			_bNext = false;
			for (int32 Index = 0; !_bNext && Index < N; ++Index)
			{
				_bNext = Chars[Index] == Char && ++Cursor;
			}
			bValid |= _bNext;
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool Token(const TCHAR* String, int32 N, FVActParseCursor& Cursor)
	{
		bool bValid = Cursor.IsValid();
		const TCHAR* _To = Cursor.To;

		for (int32 Index = 0; bValid && Index < N; ++Index)
		{
			bValid = String[Index] == *Cursor && ++Cursor;
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool TokenRange(const TCHAR& From, const TCHAR& To, FVActParseCursor& Cursor)
	{
		bool _bNext = Cursor.IsValid(), bValid = false;
		const TCHAR* _To = Cursor.To;

		while (_bNext)
		{
			const TCHAR& Char = *Cursor;
			_bNext = Char >= From && Char <= To && ++Cursor;
			bValid |= _bNext;
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool TokenContext(const TCHAR* Chars, int32 N, const TCHAR& Escape, FVActParseCursor& Cursor)
	{
		bool _bNext = Cursor.IsValid() && N > 0, bValid = false, bEscape = false;
		const TCHAR* _To = Cursor.To;
		TCHAR _Token = '\0';

		if (_bNext)
		{
			_bNext = false;
			for (int32 Index = 0; !_bNext && Index < N; ++Index)
			{
				_bNext |= (_Token = Chars[Index]) == *Cursor && ++Cursor;
			}

			while (_bNext && !bValid)
			{
				const TCHAR& _Char = *Cursor;
				bValid = !bEscape && _Token == _Char;
				bEscape = _Char == Escape && !Escape;
				++Cursor; // It may incement once to much
			}
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool TokenContext(const TCHAR& Char, const TCHAR& Escape, FVActParseCursor& Cursor)
	{
		bool _bNext = Cursor.IsValid() && Char == *Cursor && ++Cursor, bValid = false, bEscape = false;
		const TCHAR* _To = Cursor.To;

		while (_bNext && !bValid)
		{
			const TCHAR& _Char = *Cursor;
			bValid = !bEscape && Char == _Char;
			bEscape = _Char == Escape && !Escape;
			++Cursor; // It may incement once to much
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool TokenSymbol(const TCHAR& Char, const TCHAR& Escape, FVActParseCursor& Cursor)
	{
		bool _bNext = Cursor.IsValid() && Char == *Cursor && ++Cursor;
		const bool bEscape = _bNext && Escape == *Cursor && ++Cursor;
		bool bValid = _bNext && Char != *Cursor && ++Cursor;
		const TCHAR* _To = Cursor.To;

		while (bValid && _bNext && bEscape)
		{
			const TCHAR& _Char = *Cursor;
			_bNext = _Char != Char;
			bValid = _Char != Escape;
			++Cursor; // It may incement once to much
		}
		
		bValid &= !_bNext || (Char == *Cursor && ++Cursor);

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool TokenRanges(const TCHAR* Ranges, int32 N, FVActParseCursor& Cursor)
	{
		bool bValid = false;
		bool _bNext = Cursor.IsValid() && !(N % 2);
		const TCHAR* _To = Cursor.To;
		const int32 _N = N / 2;
		
		while (_bNext)
		{
			const TCHAR& Char = *Cursor;
			_bNext = false;
			for (int32 Index = 0; !_bNext && Index < _N; ++Index)
			{
				_bNext = Char >= Ranges[Index * 2] && Char <= Ranges[Index * 2 + 1] && ++Cursor;
			}
			bValid |= _bNext;
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	static bool TokenRangesNot(const TCHAR* Ranges, int32 N, FVActParseCursor& Cursor)
	{
		bool bValid = false;
		bool _bNext = Cursor.IsValid() && !(N % 2);
		const TCHAR* _To = Cursor.To;
		const int32 _N = N / 2;

		while (_bNext)
		{
			const TCHAR& Char = *Cursor;
			_bNext = false;
			for (int32 Index = 0; !_bNext && Index < _N; ++Index)
			{
				_bNext = Char <= Ranges[Index * 2] && Char >= Ranges[Index * 2 + 1] && ++Cursor;
			}
			bValid |= _bNext;
		}

		if (!bValid) { Cursor.To = _To; }
		return bValid;
	}

	FORCEINLINE static bool TokenContext(const FString& Chars, const TCHAR& Escape, FVActParseCursor& Cursor)
	{
		return TokenContext(*Chars, Chars.Len(), Escape, Cursor);
	}

	FORCEINLINE static bool TokenSkipAny(const FString& Chars, FVActParseCursor& Cursor)
	{
		return TokenSkipAny(*Chars, Chars.Len(), Cursor);
	}

	FORCEINLINE static bool TokenNot(const TCHAR& Char, FVActParseCursor& Cursor)
	{
		return *Cursor != Char && ++Cursor;
	}

	FORCEINLINE static bool TokenRanges(const FString& String, FVActParseCursor& Cursor)
	{
		return TokenRanges(*String, String.Len(), Cursor);
	}

	FORCEINLINE static bool TokenRangesNot(const FString& String, FVActParseCursor& Cursor)
	{
		return TokenRanges(*String, String.Len(), Cursor);
	}

	FORCEINLINE static bool Token(const FString& String, FVActParseCursor& Cursor)
	{
		return Token(*String, String.Len(), Cursor);
	}

	FORCEINLINE static bool Token(const TCHAR& Char, FVActParseCursor& Cursor)
	{
		return *Cursor == Char && ++Cursor;
	}

	FORCEINLINE static bool TokenSkip(int32 Count, FVActParseCursor& Cursor)
	{
		Cursor += Count;

		return true;
	}

protected:
	static bool Evaluate(bool bValid, FVActParseCursor& Cursor, FVActParseCursor& _Cursor, TArray<FVActParseCursor>& Cursors, EVActParseToken Token, int32 UndoCursors = 0)
	{
		if (bValid)
		{
			Cursor.Continue(_Cursor);
			_Cursor.Close(Token);
			_Cursor.Id = Cursors.Num();
			Cursors.Add(_Cursor);
		}
		else if (UndoCursors > 0) { Cursors.SetNum(_Cursor.Id, EAllowShrinking::No); }

		return bValid;
	}

	FVActTokenUtils() {}
};

struct VACTFILES_API FVActTextTokenUtils
	: public FVActTokenUtils
{
	static const FString TOKEN_SKIP_SET;

	static const FString TOKEN_INF;

	static const FString TOKEN_NAN;

	static const FString TOKEN_TRUE;

	static const FString TOKEN_FALSE;

	static const FString TOKEN_NULL;

	static const FString TOKEN_NAME_RANGES;

	static const FString TOKEN_HEXA_RANGES;


	static const TCHAR TOKEN_STRING;

	static const TCHAR TOKEN_CHAR;

	static const TCHAR TOKEN_HEX;

	static const TCHAR TOKEN_BLOB;

	static const TCHAR TOKEN_REF;

	static const TCHAR TOKEN_ESC;

	static const TCHAR TOKEN_SCI;


	static const TCHAR TOKEN_STRUCT_OPEN;

	static const TCHAR TOKEN_STRUCT_CLOSE;

	static const TCHAR TOKEN_ARRAY_OPEN;

	static const TCHAR TOKEN_ARRAY_CLOSE;

	static const TCHAR TOKEN_TUPLE_OPEN;

	static const TCHAR TOKEN_TUPLE_CLOSE;

	static const TCHAR TOKEN_TAG_OPEN;

	static const TCHAR TOKEN_TAG_CLOSE;

	static const TCHAR TOKEN_DILIM;

	static const TCHAR TOKEN_PROP;

	static const TCHAR TOKEN_ATTR;


	static bool TokenValue(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenKey(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenName(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenNull(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenChar(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenString(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenBlob(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenRef(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenBoolean(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenNumber(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenHexa(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

protected:
	FVActTextTokenUtils() {}
};

struct FVActBinaryTokenUtils
	: public FVActTokenUtils
{
protected:
	FVActBinaryTokenUtils() {}
};

struct VACTFILES_API FVactTextParseUtils
{
	static bool Parse(FString& Into, FVActParseRootIt& It)
	{
		const FVActParseCursor& Cursor = *It;
		bool bValid = It && EVActParseToken::String == It.Token();
		if (bValid) { Into = FString(Cursor.Len() - 2, Cursor.From + 1); }
		return bValid && ++It;
	}

	static bool Parse(FName& Into, FVActParseRootIt& It)
	{
		const FVActParseCursor& Cursor = *It;
		bool bValid = It && EVActParseToken::Name == It.Token();
		if (bValid) { Into = FName(Cursor.Len(), Cursor.From); }
		return bValid && ++It;
	}

	static bool Parse(bool& Into, FVActParseRootIt& It)
	{
		const FStringView View = (*It).View();
		bool bValid = It && EVActParseToken::Bool == It.Token();
		if (bValid) { LexFromString(Into, View); }
		return bValid && ++It;
	}

	static bool Parse(int32& Into, FVActParseRootIt& It)
	{
		const FStringView View = (*It).View();
		bool bValid = It && EVActParseToken::Int == It.Token();
		if (bValid) { LexFromString(Into, View); }
		return bValid && ++It;
	}

	static bool Parse(double& Into, FVActParseRootIt& It)
	{
		const FStringView View = (*It).View();
		bool bValid = It && EVActParseToken::Num == It.Token() || EVActParseToken::Sci == It.Token() || EVActParseToken::Int == It.Token();
		if (bValid) { LexFromString(Into, View); }
		return bValid && ++It;
	}

	static bool Parse(float& Into, FVActParseRootIt& It)
	{
		const FStringView View = (*It).View();
		bool bValid = It && EVActParseToken::Num == It.Token() || EVActParseToken::Sci == It.Token() || EVActParseToken::Int == It.Token();
		if (bValid) { LexFromString(Into, View); }
		return bValid && ++It;
	}

	static bool Parse(FVector& Into, FVActParseRootIt& It)
	{
		return It && It.Token() == EVActParseToken::_Tuple && ++It
			&& Parse(Into.X, It)
			&& Parse(Into.Y, It)
			&& Parse(Into.Z, It)
			&& It.Token() == EVActParseToken::Tuple && ++It;
	}

	static bool Parse(FQuat& Into, FVActParseRootIt& It)
	{
		return It && It.Token() == EVActParseToken::_Tuple && ++It
			&& Parse(Into.X, It)
			&& Parse(Into.Y, It)
			&& Parse(Into.Z, It)
			&& Parse(Into.W, It)
			&& It.Token() == EVActParseToken::Tuple && ++It;
	}

	static bool Parse(FVector2D& Into, FVActParseRootIt& It)
	{
		return It && It.Token() == EVActParseToken::_Tuple && ++It
			&& Parse(Into.X, It)
			&& Parse(Into.Y, It)
			&& It.Token() == EVActParseToken::Tuple && ++It;
	}

	static bool Parse(FTransform& Into, FVActParseRootIt& It)
	{
		FVector Location, Scale;
		FQuat Quaternion;

		const bool bValid = It && It.Token() == EVActParseToken::_Tuple && ++It
			&& Parse(Location, It)
			&& Parse(Quaternion, It)
			&& Parse(Scale, It)
			&& It.Token() == EVActParseToken::Tuple && ++It;

		Into.SetComponents(Quaternion, Location, Scale);

		return bValid;
	}

	static bool ParsePlace(FString& Into, FVActParseRootIt& It, uint8& Flag)
	{
		Flag = 0;
		const FVActParseCursor& Cursor = *It;
		bool bValid = It && EVActParseToken::String == It.Token();
		if (bValid) { new (&Into) FString(Cursor.Len() - 2, Cursor.From + 1); Flag = 1; }
		return bValid && ++It;
	}

	static bool ParsePlace(FName& Into, FVActParseRootIt& It, uint8& Flag)
	{
		Flag = 0;
		const FVActParseCursor& Cursor = *It;
		bool bValid = It && EVActParseToken::Name == It.Token();
		if (bValid) { new (&Into) FName(Cursor.Len(), Cursor.From); Flag = 1; }
		return bValid && ++It;
	}

	template<typename T>
	static bool ParseTuplePlace(TArray<T>& Into, FVActParseRootIt& It, uint8& Flag, int32 Stride = 0)
	{
		Flag = 0;
		bool bValid = It && It.Token() == EVActParseToken::_Tuple && ++It;
		if (bValid) { new (&Into) TArray<T>(); Flag = 1; }
		bool bNext = It.Token() != EVActParseToken::Tuple;
		while (bValid && bNext && It)
		{
			T Value;
			bValid &= Parse(Value, It += Stride);
			if (bValid) { Into.Add(Value); }
			bNext = It.Token() != EVActParseToken::Tuple;
		}
		bValid &= It.Token() == EVActParseToken::Tuple && ++It;
		if (!bValid && Flag) { Into.~TArray<T>(); Flag = 0; }
		return bValid;
	}

	template<typename T>
	static bool ParseEnum(T& Into, FVActParseRootIt& It, TMap<FName, T> Map)
	{
		const FVActParseCursor& Cursor = *It;
		bool bValid = It && EVActParseToken::Name == It.Token();
		if (bValid)
		{
			T* _Enum = Map.Find(FName(Cursor.Len(), Cursor.From));
			Into = _Enum ? *_Enum : (T)0;
		}
		return bValid && ++It;;
	}

	template<typename T>
	static bool ParseObject(T* Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr)
	{
		FString AssetPath;
		const bool bValid = Parse(AssetPath, It);
		if (bValid) { Into = Cast<T>(StaticLoadObject(Class, InOuter, *AssetPath)); }
		return bValid;
	}

	template<typename T>
	static bool ParseObject(TObjectPtr<T>& Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr)
	{
		FString AssetPath;
		const bool bValid = Parse(AssetPath, It);
		if (bValid) { Into = Cast<T>(StaticLoadObject(Class, InOuter, *AssetPath)); }
		return bValid;
	}

	template<typename T>
	static bool ParseClass(T& Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr)
	{
		FString AssetPath;
		const bool bValid = Parse(AssetPath, It);
		if (bValid) { Into = StaticLoadClass(Class, InOuter, *AssetPath); }
		return bValid;
	}

	template<typename T>
	static bool ParseClass(TSubclassOf<T>& Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr)
	{
		FString AssetPath;
		const bool bValid = Parse(AssetPath, It);
		if (bValid) { Into = StaticLoadClass(Class, InOuter, *AssetPath); }
		return bValid;
	}

	template<typename T>
	static bool Parse(TArrayView<T>& Into, FVActParseRootIt& It, int32 Stride = 0)
	{
		int32 Index = 0;
		bool bValid = It && It.Token() == EVActParseToken::_Tuple && ++It && Into.IsValidIndex(Index);
		bool bNext = It.Token() != EVActParseToken::Tuple;
		while (bValid && bNext && It)
		{
			bValid &= Into.IsValidIndex(Index) && Parse(Into[Index++], It += Stride);
			bool bNext = It.Token() != EVActParseToken::Tuple;
		}
		bValid &= It.Token() == EVActParseToken::Tuple && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseArray(TArray<T>& Into, FVActParseRootIt& It, int32 Stride = 0)
	{
		bool bValid = It && It.Token() == EVActParseToken::_Array && ++It;
		bool bNext = It.Token() != EVActParseToken::Array;
		while (bValid && bNext && It)
		{
			T Value;
			bValid &= Parse(Value, It += Stride);
			if (bValid) { Into.Add(Value); }
			bNext = It.Token() != EVActParseToken::Array;
		}
		bValid &= It.Token() == EVActParseToken::Array && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseTuple(TArray<T>& Into, FVActParseRootIt& It, int32 Stride = 0)
	{
		bool bValid = It && It.Token() == EVActParseToken::_Tuple && ++It;
		bool bNext = It.Token() != EVActParseToken::Tuple;
		while (bValid && bNext && It)
		{
			T Value;
			bValid &= Parse(Value, It += Stride);
			if (bValid) { Into.Add(Value); }
			bNext = It.Token() != EVActParseToken::Tuple;
		}
		bValid &= It.Token() == EVActParseToken::Tuple && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseTuple(T* Into, FVActParseRootIt& It, int32 Count, int32 IntoStride = 1)
	{
		int32 Index = 0;
		bool bValid = It && It.Token() == EVActParseToken::_Tuple && ++It;
		bool bNext = It.Token() != EVActParseToken::Tuple;
		while (bValid && bNext && It && Index < Count)
		{
			bValid &= Parse(Into[Index * IntoStride], It);
			bNext = It.Token() != EVActParseToken::Tuple;
			++Index;
		}
		bValid &= It.Token() == EVActParseToken::Tuple && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseClassArray(TArray<T>& Into, FVActParseRootIt& It, UClass *Class, UObject* InOuter = nullptr, int32 Stride = 0)
	{
		bool bValid = It && It.Token() == EVActParseToken::_Array && ++It;
		bool bNext = It.Token() != EVActParseToken::Array;
		while (bValid && bNext && It)
		{
			FString AssetPath;
			bValid &= Parse(AssetPath, It += Stride);
			if (bValid)
			{
				TSubclassOf<T> AssetClass = StaticLoadClass(Class, InOuter, *AssetPath);
				Into.Add(AssetClass);
			}
			bNext = It.Token() != EVActParseToken::Array;
		}
		bValid &= It.Token() == EVActParseToken::Array && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseClassArray(TArray<TSubclassOf<T>>& Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr, int32 Stride = 0)
	{
		bool bValid = It && It.Token() == EVActParseToken::_Array && ++It;
		bool bNext = It.Token() != EVActParseToken::Array;
		while (bValid && bNext && It)
		{
			FString AssetPath;
			bValid &= Parse(AssetPath, It += Stride);
			if (bValid)
			{
				TSubclassOf<T> AssetClass = StaticLoadClass(Class, InOuter, *AssetPath);
				Into.Add(AssetClass);
			}
			bNext = It.Token() != EVActParseToken::Array;
		}
		bValid &= It.Token() == EVActParseToken::Array && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseObjectArray(TArray<T>& Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr, int32 Stride = 0)
	{
		bool bValid = It && It.Token() == EVActParseToken::_Array && ++It;
		bool bNext = It.Token() != EVActParseToken::Array;
		while (bValid && bNext && It)
		{
			FString AssetPath;
			bValid &= Parse(AssetPath, It += Stride);
			if (bValid)
			{
				T Asset = Cast<T>(StaticLoadObject(Class, InOuter, *AssetPath));
				Into.Add(Asset);
			}
			bNext = It.Token() != EVActParseToken::Array;
		}
		bValid &= It.Token() == EVActParseToken::Array && ++It;
		return bValid;
	}

	template<typename T>
	static bool ParseObjectArray(TArray<TObjectPtr<T>>& Into, FVActParseRootIt& It, UClass* Class, UObject* InOuter = nullptr, int32 Stride = 0)
	{
		bool bValid = It && It.Token() == EVActParseToken::_Array && ++It;
		bool bNext = It.Token() != EVActParseToken::Array;
		while (bValid && bNext && It)
		{
			FString AssetPath;
			bValid &= Parse(AssetPath, It += Stride);
			if (bValid)
			{
				T* Asset = Cast<T>(StaticLoadObject(Class, InOuter, *AssetPath));
				Into.Add(Asset);
			}
			bNext = It.Token() != EVActParseToken::Array;
		}
		bValid &= It.Token() == EVActParseToken::Array && ++It;
		return bValid;
	}
protected:
	FVactTextParseUtils() {}
};

struct VACTFILES_API FVActTextEmitUtils
{
	static void Emit(const TCHAR* Value, int32 N, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Source.Append(Value, N);
		Cursors.Add(FVActParseCursor(EVActParseToken::String, _From, _From + N));
	}

	static void Emit(const FName& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		const FString _Value = Value.ToString();
		const int32 N = _Value.Len();
		Source.Append(*_Value, N);
		Cursors.Add(FVActParseCursor(EVActParseToken::Name, _From, _From + N));
	}

	static void Emit(const float& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		const FString _Value = FString::Printf(TEXT("%e"), Value);
		const int32 N = _Value.Len();
		Source.Append(*_Value, N);
		Cursors.Add(FVActParseCursor(EVActParseToken::Num, _From, _From + N));
	}

	static void Emit(const double& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		const FString _Value = FString::Printf(TEXT("%e"), Value);
		const int32 N = _Value.Len();
		Source.Append(*_Value, N);
		Cursors.Add(FVActParseCursor(EVActParseToken::Num, _From, _From + N));
	}

	static void Emit(const int32& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		const FString _Value = FString::FromInt(Value);
		const int32 N = _Value.Len();
		Source.Append(*_Value, N);
		Cursors.Add(FVActParseCursor(EVActParseToken::Int, _From, _From + N));
	}

	static void Emit(const bool& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		const FString _Value = Value ? FVActTextTokenUtils::TOKEN_TRUE : FVActTextTokenUtils::TOKEN_FALSE;
		const int32 N = _Value.Len();
		Source.Append(*_Value, N);
		Cursors.Add(FVActParseCursor(EVActParseToken::Bool, _From, _From + N));
	}

	static FORCEINLINE void Emit(const FString& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		FString _Value = FString::Printf(TEXT("\"%s\""), *Value);
		const int32 N = _Value.Len();
		Emit(*_Value, N, Cursors, Source);
	}

	template<typename T>
	static FORCEINLINE void EmitEnum(T& Enum, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, TArray<FName> Map)
	{
		if (Map.IsValidIndex((int32)Enum)) { Emit(Map[(int32)Enum], Cursors, Source); }
	}

	template<typename T>
	static FORCEINLINE void EmitEnum(T& Enum, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, TMap<T, FName> Map)
	{
		FName* NamePtr = Map.Find(Enum);
		if (NamePtr) { Emit(*NamePtr, Cursors, Source); }
	}

	void EmitProperty(const TCHAR* Key, EVActParseToken KeyToken, int32 N, const TCHAR* Value, EVActParseToken ValueToken, int32 M, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _Key = *Source + Source.Len();
		Source.Append(Key, N);
		Cursors.Add(FVActParseCursor(KeyToken, _Key, _Key + N));

		Source.AppendChar(FVActTextTokenUtils::TOKEN_PROP);

		const TCHAR* _Value = *Source + Source.Len();
		Source.Append(Value, M);
		Cursors.Add(FVActParseCursor(ValueToken, _Value, _Value + M));

		const TCHAR* _Prop = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Prop, _Key, _Prop));
	}

	static void Emit(const FQuat& Rotation, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitTupleOpen(Cursors, Source);
		FVActTextEmitUtils::Emit(Rotation.X, Cursors, Source); EmitDelimiter(Cursors, Source);
		FVActTextEmitUtils::Emit(Rotation.Y, Cursors, Source); EmitDelimiter(Cursors, Source);
		FVActTextEmitUtils::Emit(Rotation.Z, Cursors, Source); EmitDelimiter(Cursors, Source);
		FVActTextEmitUtils::Emit(Rotation.W, Cursors, Source);
		EmitTuple(Cursors, Source, _From);
	}

	static void EmitDelimiter(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		Source.AppendChar(FVActTextTokenUtils::TOKEN_DILIM);
	}

	static void Emit(const FVector& Vector, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitTupleOpen(Cursors, Source);
		FVActTextEmitUtils::Emit(Vector.X, Cursors, Source); EmitDelimiter(Cursors, Source);
		FVActTextEmitUtils::Emit(Vector.Y, Cursors, Source); EmitDelimiter(Cursors, Source);
		FVActTextEmitUtils::Emit(Vector.Z, Cursors, Source);
		EmitTuple(Cursors, Source, _From);
	}

	static void Emit(const FTransform& Transform, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitTupleOpen(Cursors, Source);
		Emit(Transform.GetLocation(), Cursors, Source); EmitDelimiter(Cursors, Source);
		Emit(Transform.GetRotation(), Cursors, Source); EmitDelimiter(Cursors, Source);
		Emit(Transform.GetScale3D(), Cursors, Source);
		EmitTuple(Cursors, Source, _From);
	}

	static void EmitStruct(TArray<FVActParseCursor>& Cursors, FStringBuilderBase& Source, const TCHAR* From)
	{

		Source.AppendChar(FVActTextTokenUtils::TOKEN_STRUCT_CLOSE);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Struct, From, _To));
	}

	static void EmitArray(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, const TCHAR* From)
	{
		Source.AppendChar(FVActTextTokenUtils::TOKEN_ARRAY_CLOSE);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Array, From, _To));
	}

	static void EmitTuple(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, const TCHAR* From)
	{
		Source.AppendChar(FVActTextTokenUtils::TOKEN_TUPLE_CLOSE);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Tuple, From, _To));
	}

	static void EmitTag(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, const TCHAR* From)
	{
		Source.AppendChar(FVActTextTokenUtils::TOKEN_TAG_CLOSE);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Tag, From, _To));
	}

	static void EmitProperty(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, const TCHAR* From)
	{
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Prop, From, _To));
	}

	static void EmitAttribute(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, const TCHAR* From)
	{
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Attr, From, _To));
	}

	static void EmitStructOpen(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Source.AppendChar(FVActTextTokenUtils::TOKEN_STRUCT_OPEN);
		Cursors.Add(FVActParseCursor(EVActParseToken::_Struct, _From, _From + 1));
	}

	static void EmitArrayOpen(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Source.AppendChar(FVActTextTokenUtils::TOKEN_ARRAY_OPEN);
		Cursors.Add(FVActParseCursor(EVActParseToken::_Array, _From, _From + 1));
	}

	static void EmitTupleOpen(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Source.AppendChar(FVActTextTokenUtils::TOKEN_TUPLE_OPEN);
		Cursors.Add(FVActParseCursor(EVActParseToken::_Tuple, _From, _From + 1));
	}

	static void EmitTagOpen(TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Source.AppendChar(FVActTextTokenUtils::TOKEN_TAG_OPEN);
		Cursors.Add(FVActParseCursor(EVActParseToken::_Tag, _From, _From + 1));
	}

	static void EmitPropertyOpen(const FName& Key, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_PROP);
	}

	static void EmitAttributeOpen(const FName& Key, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_ATTR);
	}

	template<typename T>
	static void EmitEnumProperty(const FName& Key, const T& Enum, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, TArray<FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_PROP);
		EmitEnum(Enum, Cursors, Source, Map);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Prop, _From, _To));
	}

	template<typename T>
	static void EmitEnumProperty(const FName& Key, const T& Enum, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, TMap<T, FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_PROP);
		EmitEnum(Enum, Cursors, Source, Map);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Prop, _From, _To));
	}

	template<typename T>
	static void EmitProperty(const FName& Key, const T& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_PROP);
		Emit(Value, Cursors, Source);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Prop, _From, _To));
	}

	template<typename T>
	static void EmitAttribute(const FName& Key, const T& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_ATTR);
		Emit(Value, Cursors, Source);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Attr, _From, _To));
	}

	template<typename T>
	static void EmitAttribute(const FName& Key, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Attr, _From, _To));
	}

	template<typename T>
	static void EmitEnumAttribute(const FName& Key, const T& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, TArray<FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_ATTR);
		EmitEnum(Value, Cursors, Source, Map);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Attr, _From, _To));
	}

	template<typename T>
	static void EmitEnumAttribute(const FName& Key, const T& Value, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source, TMap<T, FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		Emit(Key, Cursors, Source);
		Source.AppendChar(FVActTextTokenUtils::TOKEN_ATTR);
		EmitEnum(Value, Cursors, Source, Map);
		const TCHAR* _To = *Source + Source.Len();
		Cursors.Add(FVActParseCursor(EVActParseToken::Attr, _From, _To));
	}

	template<typename T>
	static void EmitTuple(T* Tuple, TArray<FVActParseCursor>& Cursors, int32 Count, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitTupleOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			Emit(Tuple[Index], Cursors, Source);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitTuple(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitTuple(TArray<T> Tuple, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const int32 Count = Tuple.Num();
		const TCHAR* _From = *Source + Source.Len();
		EmitTupleOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Emit(Tuple[Index], Cursors, Source);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitTuple(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitArray(T* Array, TArray<FVActParseCursor>& Cursors, int32 Count, TStringBuilderBase<TCHAR>& Source)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			Emit(Array[Index], Cursors, Source); 
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitEnumArray(T* Array, TArray<FVActParseCursor>& Cursors, int32 Count, TStringBuilderBase<TCHAR>& Source, TArray<FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			EmitEnum(Array[Index], Cursors, Source, Map);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitEnumArray(T* Array, TArray<FVActParseCursor>& Cursors, int32 Count, TStringBuilderBase<TCHAR>& Source, TMap<T, FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			EmitEnum(Array[Index], Cursors, Source, Map);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitArray(TArray<T> Array, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const int32 Count = Array.Num();
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Emit(Array[Index], Cursors, Source);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitEnumArray(TArray<T> Array, TArray<FVActParseCursor>& Cursors, int32 Count, TStringBuilderBase<TCHAR>& Source, TArray<FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{
			EmitEnum(Array[Index], Cursors, Source, Map);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitEnumArray(TArray<T> Array, TArray<FVActParseCursor>& Cursors, int32 Count, TStringBuilderBase<TCHAR>& Source, TMap<T, FName> Map)
	{
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{
			EmitEnum(Array[Index], Cursors, Source, Map);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitRefArray(TArray<T> Array, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const int32 Count = Array.Num();
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{
			Emit(Array[Index]->GetPathName(), Cursors, Source);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

	template<typename T>
	static void EmitRefArray(TArray<TObjectPtr<T>> Array, TArray<FVActParseCursor>& Cursors, TStringBuilderBase<TCHAR>& Source)
	{
		const int32 Count = Array.Num();
		const TCHAR* _From = *Source + Source.Len();
		EmitArrayOpen(Cursors, Source);
		for (int32 Index = 0; Index < Count; ++Index)
		{ 
			Emit(Array[Index]->GetPathName(), Cursors, Source);
			if (Index < Count - 1) { EmitDelimiter(Cursors, Source); }
		}
		EmitArray(Cursors, Source, _From);
	}

protected:
	FVActTextEmitUtils() {}
};