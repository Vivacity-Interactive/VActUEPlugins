#pragma once

#include "UObject/Package.h"
#include "OICProfile.h"

#include "Components/SceneComponent.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "VActOICEditor.generated.h"

enum class ECmdVActExportOptions
{
	None = 0,
	PerLevel = 1,
	PerWorld = 2,
	AllCombined = 3,
};

USTRUCT()
struct VACTOICEDITOR_API FVActOICEditor
{
	GENERATED_BODY()

	static const TMap<FString, ECmdVActExportOptions> ExportOptionNameToEnum;

public:
	static const TArray<FString> ExportOptionNames;

	static void Cmd_ExportToOICAsset(const TArray<FString>& Args);

	static void ExportToOICAsset(const UWorld* World, const FString& Path, ECmdVActExportOptions Options = ECmdVActExportOptions::AllCombined);

protected:
	static void _ResolveSelected(TArray<FAssetData>& Assets);

	static void _SaveOICProfileAsset(UOICProfile* Profile, UPackage* Package);

};