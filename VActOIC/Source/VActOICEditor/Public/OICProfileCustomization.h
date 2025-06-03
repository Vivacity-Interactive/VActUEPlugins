#pragma once

#include "CoreMinimal.h"
#include "VActFileTypes.h"

#include "OICProfile.h"

#include "VActFileFromToCustomization.h"

class FOICProfileCustomization : public FVActFileFromToCustomization<UOICProfile, FOICProfileCustomization>
{
public:
	static const FString TypeFilter;

	static const FName EditCategory;

	static const TArray<FName> PropertyNames;

	static TSharedRef<IDetailCustomization> MakeInstance();
	
	static bool ParseFromCursorsBinary(class UOICProfile* InContext, const FVActParseRoot& Root);

	static bool ParseFromCursorsCompact(class UOICProfile* InContext, const FVActParseRoot& Root);

	static bool ParseFromCursorsJson(class UOICProfile* InContext, const FVActParseRoot& Root);

	static bool ParseFromCursorsJsonStrict(class UOICProfile* InContext, const FVActParseRoot& Root);

	static bool ComposeToCursors(class UOICProfile* InContext, FVActComposeRoot& Root);

	static bool ComposeToCursorsCompact(class UOICProfile* InContext, FVActComposeRoot& Root);

protected:
	static bool _ParseOICMeta(FOICMeta& Into, FVActParseRootIt& It, UObject* InOuter = nullptr);

};