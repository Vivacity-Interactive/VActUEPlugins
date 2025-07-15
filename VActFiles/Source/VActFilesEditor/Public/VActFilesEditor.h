#pragma once

#include "CoreMinimal.h"
#include "AssetRegistry/AssetData.h"
#include "VActFilesEditor.generated.h"

enum class ECmdVActExportOptions
{
	None = 0,
	PromptAll = 1,
	NoPrompt = 2,
	PromptOnce = 3
};

USTRUCT()
struct VACTFILESEDITOR_API FVActFilesEditor
{
	GENERATED_BODY()

	static const TMap<FString, ECmdVActExportOptions> ExportOptionNameToEnum;

public:
	static const TArray<FString> ExportOptionNames;
	
	static void Cmd_ExportSelectedStaticMeshes(const TArray<FString>& Args);

	static void ExportStaticMeshes(TArray<FAssetData>& Assets, const FString& Path, ECmdVActExportOptions Options = ECmdVActExportOptions::PromptAll);

	static void Cmd_ExportSelectedTextures(const TArray<FString>& Args);

	static void ExportTextures(TArray<FAssetData>& Assets, const FString& Path, ECmdVActExportOptions Options = ECmdVActExportOptions::PromptAll);

protected:
	static void _ResolveSelected(TArray<FAssetData>& Assets);
};