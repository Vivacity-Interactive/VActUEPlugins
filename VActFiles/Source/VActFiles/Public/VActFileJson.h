#pragma once

#include "VActFileTypes.h"
#include "VActFileUtils.h"

struct VACTFILES_API FVActFileJson
	: public FVActTextTokenUtils
{
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


	static bool Load(FVActParseRoot& Root, const TCHAR* Path);

	static bool Save(const FVActComposeRoot& Root, const TCHAR* Path);


	static bool TokenRoot(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenCollection(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenStruct(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenArray(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenTuple(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenTag(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenProperty(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

	static bool TokenNamed(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors);

protected:
	FVActFileJson() {};

};