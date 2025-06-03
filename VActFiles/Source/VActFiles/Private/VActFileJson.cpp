#include "VActFileJson.h"

#include "HAL//PlatformFileManager.h"

const TCHAR FVActFileJson::TOKEN_STRUCT_OPEN = '{';

const TCHAR FVActFileJson::TOKEN_STRUCT_CLOSE = '}';

const TCHAR FVActFileJson::TOKEN_ARRAY_OPEN = '[';

const TCHAR FVActFileJson::TOKEN_ARRAY_CLOSE = ']';

const TCHAR FVActFileJson::TOKEN_TUPLE_OPEN = '(';

const TCHAR FVActFileJson::TOKEN_TUPLE_CLOSE = ')';

const TCHAR FVActFileJson::TOKEN_TAG_OPEN = '<';

const TCHAR FVActFileJson::TOKEN_TAG_CLOSE = '>';

const TCHAR FVActFileJson::TOKEN_DILIM = ',';

const TCHAR FVActFileJson::TOKEN_PROP = ':';


bool FVActFileJson::Load(FVActParseRoot& Root, const TCHAR* Path)
{
	FVActParseCursor Cursor;

	const bool bValid = FPlatformFileManager::Get().GetPlatformFile().FileExists(Path)
		&& FFileHelper::LoadFileToString(Root.Source, Path)
		&& TokenRoot(Cursor.Init(Root.Source), Root.Cursors);
	
	return bValid;
}

bool FVActFileJson::Save(const FVActComposeRoot& Root, const TCHAR* Path)
{
	for (const FVActComposeCursor& Cursor : Root.Cursors)
	{
#if WITH_EDITORONLY_DATA
		const bool _b_Index = _DEBUG_VActParseInfo::TokenNames.IsValidIndex((int32)Cursor.Token);
		UE_LOG(LogTemp, Log, TEXT("%7s(%d)"), (_b_Index ? *_DEBUG_VActParseInfo::TokenNames[(int32)Cursor.Token] : TEXT("")), (int32)Cursor.Token);
#endif
	}

	return false;
}

bool FVActFileJson::TokenRoot(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	TokenSkipAny(TOKEN_SKIP_SET, Cursor);

	return TokenCollection(Cursor, Cursors)
		//|| TokenNamed(Cursor, Cursor) // TODO problem consumes special numbers it should to a left look 
		|| TokenValue(Cursor, Cursors);
}

bool FVActFileJson::TokenCollection(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	return TokenStruct(Cursor, Cursors)
		|| TokenArray(Cursor, Cursors)
		|| TokenTuple(Cursor, Cursors)
		|| TokenTag(Cursor, Cursors);
}

bool FVActFileJson::TokenStruct(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	bool bValid = Token(TOKEN_STRUCT_OPEN, _Cursor), _bValid = bValid;
	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);

	if (bValid) { Cursors.Add(FVActParseCursor(_Cursor, EVActParseToken::_Struct)); }

	bool bNext = true;
	while (_bValid && bNext)
	{
		_bValid = TokenProperty(_Cursor, Cursors) && TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
		bNext = Token(',', _Cursor) && TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	}

	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	bValid &= !bNext && Token(TOKEN_STRUCT_CLOSE, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Struct);
}

bool FVActFileJson::TokenArray(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	bool bValid = Token(TOKEN_ARRAY_OPEN, _Cursor), _bValid = bValid;
	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);

	if (bValid) { Cursors.Add(FVActParseCursor(_Cursor, EVActParseToken::_Array)); }

	bool bNext = true;
	while (_bValid && bNext)
	{
		_bValid = TokenRoot(_Cursor, Cursors) && TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
		bNext = Token(',', _Cursor) && TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	}

	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	bValid &= !bNext && Token(TOKEN_ARRAY_CLOSE, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Array);
}

bool FVActFileJson::TokenTuple(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	bool bValid = Token(TOKEN_TUPLE_OPEN, _Cursor), _bValid = bValid;
	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);

	if (bValid) { Cursors.Add(FVActParseCursor(_Cursor, EVActParseToken::_Tuple)); }

	bool bNext = true;
	while (_bValid && bNext)
	{
		_bValid = TokenRoot(_Cursor, Cursors) 
			&& TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
		bNext = Token(',', _Cursor)
			&& TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	}

	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	bValid &= !bNext && Token(TOKEN_TUPLE_CLOSE, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Tuple);
}

bool FVActFileJson::TokenTag(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	bool bValid = Token(TOKEN_TAG_OPEN, _Cursor), _bValid = bValid;
	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);

	if (bValid) { Cursors.Add(FVActParseCursor(_Cursor, EVActParseToken::_Tag)); }

	bool bNext = true;
	while (_bValid && bNext)
	{
		_bValid = TokenRoot(_Cursor, Cursors)
			&& TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
		bNext = Token(',', _Cursor)
			&& TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	}

	TokenSkipAny(TOKEN_SKIP_SET, _Cursor);
	bValid &= !bNext && Token(TOKEN_TAG_CLOSE, _Cursor);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Tag);
}

bool FVActFileJson::TokenProperty(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open();

	const bool bValid = TokenKey(_Cursor, Cursors)
		&& TokenSkipAny(TOKEN_SKIP_SET, _Cursor)
		&& Token(TOKEN_PROP, _Cursor)
		&& TokenSkipAny(TOKEN_SKIP_SET, _Cursor)
		&& TokenRoot(_Cursor, Cursors);

	return Evaluate(bValid, Cursor, _Cursor, Cursors, EVActParseToken::Prop);
}

bool FVActFileJson::TokenNamed(FVActParseCursor& Cursor, TArray<FVActParseCursor>& Cursors)
{
	FVActParseCursor _Cursor = Cursor.Open(), _Cursor2;;
	EVActParseToken _Token = EVActParseToken::Name;
	bool bValid = false;

	const bool bName = TokenName(_Cursor, Cursors);
	if (bName) {
		_Cursor2 = FVActParseCursor(_Cursor);

		bValid = TokenString(_Cursor2, Cursors) || TokenTuple(_Cursor2, Cursors);
		if (bValid)
		{
			_Cursor.Set(_Cursor2);
			_Token = _Cursor2.Token == EVActParseToken::String ? EVActParseToken::XString : EVActParseToken::XTuple;
		}
	}

	return Evaluate(bValid, Cursor, _Cursor, Cursors, _Token) || bName;
}