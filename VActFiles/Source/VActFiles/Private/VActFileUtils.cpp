#include "VActFileUtils.h"

const TArray<TSharedPtr<FString>> FVActFileUtils::FormatOptions = {
	MakeShared<FString>(TEXT("None")),
	MakeShared<FString>(TEXT("Json")),
	MakeShared<FString>(TEXT("Compact")),
	MakeShared<FString>(TEXT("Binary"))
};

const TMap<TSharedPtr<FString>, EVActFileFormat> FVActFileUtils::MapFormat = {
	{ FormatOptions[(int32)EVActFileFormat::None], EVActFileFormat::None },
	{ FormatOptions[(int32)EVActFileFormat::Json], EVActFileFormat::Json },
	{ FormatOptions[(int32)EVActFileFormat::Compact], EVActFileFormat::Compact },
	{ FormatOptions[(int32)EVActFileFormat::Binary], EVActFileFormat::Binary }
};

#if WITH_EDITORONLY_DATA
const TArray<FString> _DEBUG_VActParseInfo::TokenNames = {
	FString(TEXT("None")),
	FString(TEXT("Prop")),
	FString(TEXT("Join")),
	FString(TEXT("Name")),
	FString(TEXT("String")),
	FString(TEXT("Blob")),
	FString(TEXT("Struct")),
	FString(TEXT("Array")),
	FString(TEXT("Tuple")),
	FString(TEXT("Record")),
	FString(TEXT("Tag")),
	FString(TEXT("Bool")),
	FString(TEXT("Num")),
	FString(TEXT("Sci")),
	FString(TEXT("Int")),
	FString(TEXT("Hex")),
	FString(TEXT("Null")),
	FString(TEXT("Ref")),
	FString(TEXT("Char")),
	FString(TEXT("XTuple")),
	FString(TEXT("XString")),
	FString(TEXT("_Struct")),
	FString(TEXT("_Array")),
	FString(TEXT("_Tuple")),
	FString(TEXT("_Record")),
	FString(TEXT("_Tag"))
};
#endif

uint32 GetTypeHash(const TSharedPtr<FString>& Ptr)
{
	return ::PointerHash(Ptr.Get());
}

const FString FVActTextTokenUtils::TOKEN_SKIP_SET = TEXT("\n\r\t ");

const FString FVActTextTokenUtils::TOKEN_INF = TEXT("inf");

const FString FVActTextTokenUtils::TOKEN_NAN = TEXT("nan");

const FString FVActTextTokenUtils::TOKEN_TRUE = TEXT("true");

const FString FVActTextTokenUtils::TOKEN_FALSE = TEXT("false");

const FString FVActTextTokenUtils::TOKEN_NULL = TEXT("null");

const FString FVActTextTokenUtils::TOKEN_NAME_RANGES = TEXT("__azAZ09");

const FString FVActTextTokenUtils::TOKEN_HEXA_RANGES = TEXT("afAF09");

const TCHAR FVActTextTokenUtils::TOKEN_STRING = '"';

const TCHAR FVActTextTokenUtils::TOKEN_CHAR = '\'';

const TCHAR FVActTextTokenUtils::TOKEN_HEX = '#';

const TCHAR FVActTextTokenUtils::TOKEN_BLOB = '&';

const TCHAR FVActTextTokenUtils::TOKEN_REF = '@';

const TCHAR FVActTextTokenUtils::TOKEN_ESC = '\\';

const TCHAR FVActTextTokenUtils::TOKEN_SCI = 'e';


bool FVActTextTokenUtils::TokenValue(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	return TokenString(Cursor, Cursors)
		|| TokenBlob(Cursor, Cursors)
		|| TokenRef(Cursor, Cursors)
		|| TokenHexa(Cursor, Cursors)
		|| TokenChar(Cursor, Cursors)
		|| TokenBoolean(Cursor, Cursors)
		|| TokenNumber(Cursor, Cursors)
		|| TokenNull(Cursor, Cursors)
		|| TokenName(Cursor, Cursors);
}

bool FVActTextTokenUtils::TokenKey(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	return TokenNumber(Cursor, Cursors)
		|| TokenString(Cursor, Cursors)
		|| TokenName(Cursor, Cursors);
}

bool FVActTextTokenUtils::TokenName(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = TokenRanges(TOKEN_NAME_RANGES, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Name);
}

bool FVActTextTokenUtils::TokenNull(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = Token(TOKEN_NULL, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Null);
}

bool FVActTextTokenUtils::TokenChar(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = TokenSymbol(TOKEN_CHAR, TOKEN_ESC, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Char);
}

bool FVActTextTokenUtils::TokenString(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = TokenContext(TOKEN_STRING, TOKEN_ESC, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::String);
}

bool FVActTextTokenUtils::TokenBlob(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = Token(TOKEN_BLOB, _Cursor)
		&& TokenContext(TOKEN_STRING, TOKEN_ESC, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Blob);
}

bool FVActTextTokenUtils::TokenBoolean(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{

	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = Token(TOKEN_TRUE, _Cursor) || Token(TOKEN_FALSE, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Bool);
}

bool FVActTextTokenUtils::TokenNumber(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open(), _Cursor2;
	EVActParseToken _Token = EVActParseToken::Num;

	bool bInteger = false, bNumber = false;

	Token('-', _Cursor) || Token('+', _Cursor);

	bool bValid = Token(TOKEN_NAN, _Cursor) || Token(TOKEN_INF, _Cursor);

	if (!bValid)
	{
		bInteger = TokenRange('0', '9', _Cursor);

		bNumber = Token('.', _Cursor) && TokenRange('0', '9', _Cursor);

		_Token = bNumber ? EVActParseToken::Num : EVActParseToken::Int;

		bValid = bNumber || bInteger;

		if (bValid)
		{
			_Cursor2 = FVActParseCursor(_Cursor);

			bool bScientific = Token(TOKEN_SCI, _Cursor2)
				&& (Token('-', _Cursor2) || Token('+', _Cursor2))
				&& TokenRange('0', '9', _Cursor2);

			if (bScientific)
			{
				_Cursor.Set(_Cursor2);
				_Token = EVActParseToken::Sci;
			}
		}
	}

	return Evaluate(bValid, Cursor, _Cursor, Cursors, _Token);
}

bool FVActTextTokenUtils::TokenHexa(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = Token(TOKEN_HEX, _Cursor)
		&& TokenRanges(TOKEN_HEXA_RANGES, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Hex);
}

bool FVActTextTokenUtils::TokenRef(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = Token(TOKEN_REF, _Cursor)
		&& (TokenRanges(TOKEN_NAME_RANGES, _Cursor)
			|| TokenContext(TOKEN_STRING, TOKEN_ESC, _Cursor));

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Ref);
}