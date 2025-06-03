#pragma once

#include "VActFileTypes.h"

struct VACTFILES_API FVActFileBinary
{
	static bool Load(FVActParseRoot& Root, const TCHAR* Path);

	static bool Save(const FVActComposeRoot& Root, const TCHAR* Path);

private:
	FVActFileBinary() {};
};