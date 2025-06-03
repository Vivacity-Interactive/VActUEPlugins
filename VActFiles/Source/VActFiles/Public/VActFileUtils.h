#pragma once

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
};

struct VACTFILES_API FVActComposeUtils
{
	static void ComposeProperty(const void* Key, EVActParseToken KeyToken, int32 N, const void* Value, EVActParseToken ValueToken, int32 M, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(KeyToken, Key, N));
		Cursors.Add(FVActComposeCursor(ValueToken, Value, M));
		Cursors.Add(FVActComposeCursor(EVActParseToken::Prop));
	}

	static void Compose(const FQuat& Rotation, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeTupleOpen(Cursors);
		Compose(Rotation.X, Cursors);
		Compose(Rotation.Y, Cursors);
		Compose(Rotation.Z, Cursors);
		Compose(Rotation.W, Cursors);
		ComposeTuple(Cursors);
	}

	static void Compose(const FVector& Vector, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeTupleOpen(Cursors);
		Compose(Vector.X, Cursors);
		Compose(Vector.Y, Cursors);
		Compose(Vector.Z, Cursors);
		ComposeTuple(Cursors);
	}

	static void Compose(const FTransform& Transform, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeTupleOpen(Cursors);
		Compose(Transform.GetLocation(), Cursors);
		Compose(Transform.GetRotation(), Cursors);
		Compose(Transform.GetScale3D(), Cursors);
		ComposeTuple(Cursors);
	}

	template<typename T>
	static void ComposeEnum(T& Enum, TArray<FVActComposeCursor>& Cursors, TArray<FName> Map)
	{
		if (Map.IsValidIndex((int32)Enum)) { FVActComposeUtils::Compose(Map[(int32)Enum], Cursors); }
	}

	template<typename T>
	static void ComposeEnum(T& Enum, TArray<FVActComposeCursor>& Cursors, TMap<T, FName> Map)
	{
		FName* NamePtr = Map.Find(Enum);
		if (NamePtr) { FVActComposeUtils::Compose(*NamePtr, Cursors); }
	}

	template<typename T>
	static void ComposeTuple(T* Tuple, TArray<FVActComposeCursor>& Cursors, int32 Count)
	{
		ComposeTupleOpen(Cursors);
		for (int32 Index = 0; Index < Count; ++Index) { FVActComposeUtils::Compose(Tuple[Index], Cursors); }
		ComposeTuple(Cursors);
	}

	template<typename T>
	static void ComposeTuple(TArray<T> Tuple, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeTupleOpen(Cursors);
		for (const T& _Ref : Tuple) { FVActComposeUtils::Compose(_Ref, Cursors); }
		ComposeTuple(Cursors);
	}

	template<typename T>
	static void ComposeArray(T* Array, TArray<FVActComposeCursor>& Cursors, int32 Count)
	{
		ComposeArrayOpen(Cursors);
		for (int32 Index = 0; Index < Count; ++Index) { FVActComposeUtils::Compose(Array[Index], Cursors); }
		ComposeArray(Cursors);
	}

	template<typename T>
	static void ComposeArray(TArray<T> Array, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeArrayOpen(Cursors);
		for (const T& _Ref : Array) { FVActComposeUtils::Compose(_Ref, Cursors); }
		ComposeArray(Cursors);
	}

	template<typename T>
	static void ComposeRefArray(TArray<T> Array, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeArrayOpen(Cursors);
		for (const T& _Ref : Array) { FVActComposeUtils::Compose(_Ref->GetPathName(), Cursors); }
		ComposeArray(Cursors);
	}

	template<typename T>
	static void ComposeRefArray(TArray<TObjectPtr<T>> Array, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeArrayOpen(Cursors);
		for (const TObjectPtr<T>& _Ref : Array) { FVActComposeUtils::Compose(_Ref->GetPathName(), Cursors); }
		ComposeArray(Cursors);
	}

	static FORCEINLINE void ComposeStruct(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Struct));
	}

	static FORCEINLINE void ComposeArray(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Array));
	}

	static FORCEINLINE void ComposeTuple(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Tuple));
	}

	static FORCEINLINE void ComposeRecord(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Record));
	}

	static FORCEINLINE void ComposeProperty(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Prop));
	}

	static FORCEINLINE void ComposeStructOpen(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::_Struct));
	}

	static FORCEINLINE void ComposeArrayOpen(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::_Array));
	}

	static FORCEINLINE void ComposeTupleOpen(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::_Tuple));
	}

	static FORCEINLINE void ComposeRecordOpen(TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::_Record));
	}

	static FORCEINLINE void Compose(const TCHAR* Value, int32 N, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::String, (void*)Value, N));
	}

	static FORCEINLINE void Compose(const FName& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Name, (void*)&Value, sizeof(FName)));
	}

	static FORCEINLINE void Compose(const float& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Num, (void*)&Value, sizeof(float)));
	}

	static FORCEINLINE void Compose(const double& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Num, (void*)&Value, sizeof(double)));
	}

	static FORCEINLINE void Compose(const int32& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Int, (void*)&Value, sizeof(int32)));
	}

	static FORCEINLINE void Compose(const bool& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Cursors.Add(FVActComposeCursor(EVActParseToken::Bool, (void*)&Value, sizeof(bool)));
	}

	static FORCEINLINE void ComposeProperty(const FName& Key, const int32& Value, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeProperty(&Key, EVActParseToken::Name, sizeof(FName), &Value, EVActParseToken::Int, sizeof(FName), Cursors);
	}

	static FORCEINLINE void ComposeProperty(const FName& Key, const float& Value, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeProperty(&Key, EVActParseToken::Name, sizeof(FName), &Value, EVActParseToken::Num, sizeof(FName), Cursors);
	}

	static FORCEINLINE void ComposeProperty(const FName& Key, const FName& Value, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeProperty(&Key, EVActParseToken::Name, sizeof(FName), &Value, EVActParseToken::Name, sizeof(FName), Cursors);
	}

	static FORCEINLINE void ComposeProperty(const FName& Key, const FString& Value, TArray<FVActComposeCursor>& Cursors)
	{
		ComposeProperty(&Key, EVActParseToken::Name, sizeof(FName), *Value, EVActParseToken::String, Value.Len() * sizeof(TCHAR), Cursors);
	}

	static FORCEINLINE void Compose(const FString& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Compose(*Value, Value.Len() * sizeof(TCHAR), Cursors);
	}

	static FORCEINLINE void TokenName(const FString& Value, TArray<FVActComposeCursor>& Cursors)
	{
		Compose(*Value, Value.Len() * sizeof(TCHAR), Cursors);
	}
};