#pragma once

#include "VActFileTypes.h"
#include "VActFileUtils.h"

struct VACTFILES_API FVActFileCompact
	: public FVActTextTokenUtils
{
	static bool Load(FVActParseRoot& Root, const TCHAR* Path);

	static bool Save(const FVActComposeRoot& Root, const TCHAR* Path);

protected:
	FVActFileCompact() {};
};