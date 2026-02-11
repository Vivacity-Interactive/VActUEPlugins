#pragma once

#include "VActFileTypes.h"
#include "VActFileUtils.h"

struct VACTFILES_API FVActFileJson
	: public FVActTextTokenUtils
{
	static bool Load(FVActParseRoot& Root, const TCHAR* Path);

	static bool Save(FVActEmitRoot& Root, const TCHAR* Path);


	// Tokens

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