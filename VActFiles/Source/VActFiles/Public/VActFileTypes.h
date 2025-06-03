#pragma once

#include "VActFileEntryTypes.h"

enum class EVActParseToken
{
	None = 0,
	Prop,
	Join,
	Name,
	String,
	Blob,
	Struct,
	Array,
	Tuple,
	Record,
	Tag,
	Bool,
	Num,
	Sci,
	Int,
	Hex,
	Null,
	Ref,
	Char,
	XTuple,
	XString,
	_Struct,
	_Array,
	_Tuple,
	_Record,
	_Tag
};

struct VACTFILES_API FVActComposeCursor
{
	EVActParseToken Token;
	SIZE_T Size;
	const void* Data;

	FVActComposeCursor(EVActParseToken InToken, const void* InData, SIZE_T InSize)
	{
		Token = InToken;
		Data = nullptr;
		Size = 0;
	}

	FVActComposeCursor(EVActParseToken InToken = EVActParseToken::None) : FVActComposeCursor(InToken, nullptr, 0)
	{

	}

	FORCEINLINE FVActComposeCursor& Init(const void* InData, SIZE_T InSize)
	{
		Data = InData;
		Size = InSize;
		return *this;
	}
};

struct VACTFILES_API FVActParseCursor
{
	EVActParseToken Token;
	int32 Id;
	const TCHAR* From, * To, * _End;

	FVActParseCursor()
	{
		Token = EVActParseToken::None;
		_End = To = From = nullptr;
		Id = -1;
	}

	FVActParseCursor(const FString& Data)
	{
		Init(Data, EVActParseToken::None, -1);
	}

	FVActParseCursor(const FVActParseCursor& InCursor)
	{
		Set(InCursor);
	}

	FVActParseCursor(const FVActParseCursor& InCursor, EVActParseToken InToken)
	{
		Set(InCursor);
		Token = InToken;
	}

	FVActParseCursor& Set(const FVActParseCursor& Cursor)
	{
		Token = Cursor.Token;
		From = Cursor.From;
		To = Cursor.To;
		_End = Cursor._End;
		Id = Cursor.Id;
		return *this;
	}

	FVActParseCursor& Init(const FString& Data, EVActParseToken InToken = EVActParseToken::None, int32 Offset = 0)
	{
		To = From = *Data;
		_End = From + Data.Len();
		Token = InToken;
		Id = Offset;
		return *this;
	}

	FORCEINLINE FVActParseCursor& Continue(const FVActParseCursor& Cursor)
	{
		To = Cursor.To;
		return *this;
	}

	FORCEINLINE FVActParseCursor& Reset()
	{
		To = From;
		return *this;
	}

	FORCEINLINE FVActParseCursor& Close(EVActParseToken InToken)
	{
		Token = InToken;
		return *this;
	}

	FORCEINLINE FVActParseCursor Open()
	{
		FVActParseCursor Cursor = FVActParseCursor(*this);
		Cursor.From = To;
		return Cursor;
	}

	FORCEINLINE bool IsValid()
	{
		return To <= _End;
	}

	FORCEINLINE int32 Len() const
	{
		return To - From;
	}

	FORCEINLINE FVActParseCursor& operator++()
	{
		++To;
		return *this;
	}

	FORCEINLINE FVActParseCursor& operator+=(const int32& N)
	{
		To += N;
		return *this;
	}

	FORCEINLINE const TCHAR& operator*() const
	{
		return *To;
	}

	FORCEINLINE explicit operator bool() const
	{
		return To <= _End;
	};

	FORCEINLINE FStringView View()
	{
		return FStringView(From, Len());
	};

	FORCEINLINE const FStringView View() const
	{
		return FStringView(From, Len());
	};

};

struct VACTFILES_API FVActParseRoot
{
	EVActFileFormat Format;
	TArray<FVActParseCursor> Cursors;
	FString Source;

	FVActParseRoot(EVActFileFormat InFormat = EVActFileFormat::None)
	{
		Format = InFormat;
	}
};

struct VACTFILES_API FVActParseRootIt
{
	const FVActParseRoot& Root;
	int32 Index;

	FVActParseRootIt(const FVActParseRoot& InRoot, int32 InIndex = 0) : Root(InRoot)
	{
		Index = InIndex;
	}

	FORCEINLINE bool IsValid(int32 Peek = 0)
	{
		return  Root.Cursors.IsValidIndex(Index + Peek);
	}

	FORCEINLINE EVActParseToken Token(int32 Peek = 0) const
	{
		return Root.Cursors[Index + Peek].Token;
	}

	FORCEINLINE FVActParseRootIt& operator++()
	{
		++Index;
		return *this;
	}

	FORCEINLINE FVActParseRootIt& operator+=(const int32& N)
	{
		Index += N;
		return *this;
	}

	FORCEINLINE const FVActParseCursor& operator*() const
	{
		return Root.Cursors[Index];
	}

	FORCEINLINE explicit operator bool() const
	{
		return Root.Cursors.IsValidIndex(Index);
	};
};

struct VACTFILES_API FVActComposeRoot
{
	TArray<FVActComposeCursor> Cursors;
};