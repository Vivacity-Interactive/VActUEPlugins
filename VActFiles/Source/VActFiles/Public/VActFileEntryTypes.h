#pragma once

#include "CoreMinimal.h"
#include "VActFileEntryTypes.generated.h"

UENUM()
enum class EVActFileFormat
{
	None = 0,
	Json = 1,
	Compact = 2,
	Binary = 3
};

USTRUCT()
struct VACTFILES_API FVActFileIOInfo
{
	GENERATED_BODY()

	UPROPERTY()
	bool bStrict = true;

	UPROPERTY()
	FString FromFilePath;

	UPROPERTY()
	FString FromMetaFilePath;

	UPROPERTY()
	FString ToFilePath;

	UPROPERTY()
	EVActFileFormat FromFormat;

	UPROPERTY()
	EVActFileFormat ToFormat;
};