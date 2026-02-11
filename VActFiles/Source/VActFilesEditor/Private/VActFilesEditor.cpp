#include "VActFilesEditor.h"

#include "Editor.h"
#include "ContentBrowserModule.h"
#include "AssetRegistry/AssetData.h"
#include "AssetExportTask.h"
#include "Exporters/Exporter.h"
#include "Exporters/FbxExportOption.h"
#include "IContentBrowserSingleton.h"
#include "Misc/Paths.h"

#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"

#define _VACT_FILES_EDITOR_EXPORT_MSG_SUCCESS  TEXT("Exported %s to %s")
#define _VACT_FILES_EDITOR_EXPORT_MSG_FAILED  TEXT("Failed to export %s")
#define _VACT_FILES_EDITOR_CMD_MSG_FAILED TEXT("No export arguments where provided")

const TArray<FString> FVActFilesEditor::ExportOptionNames = {
	TEXT("None"),
	TEXT("PromptAll"),
	TEXT("NoPrompt"),
	TEXT("PromptOnce")
};

const TMap<FString, ECmdVActExportOptions> FVActFilesEditor::ExportOptionNameToEnum = {
	{ ExportOptionNames[(int32)ECmdVActExportOptions::None], ECmdVActExportOptions::None},
	{ ExportOptionNames[(int32)ECmdVActExportOptions::PromptAll], ECmdVActExportOptions::PromptAll},
	{ ExportOptionNames[(int32)ECmdVActExportOptions::NoPrompt], ECmdVActExportOptions::NoPrompt},
	{ ExportOptionNames[(int32)ECmdVActExportOptions::PromptOnce], ECmdVActExportOptions::PromptOnce}
};

void FVActFilesEditor::ExportStaticMeshes(TArray<FAssetData>& Assets, const FString& Path, ECmdVActExportOptions Options)
{
	const FString Ext = TEXT(".fbx");

	bool bPrompt = false 
		|| Options == ECmdVActExportOptions::PromptAll
		|| Options == ECmdVActExportOptions::PromptOnce;

	for (const FAssetData& AssetData : Assets)
	{
		UObject* Asset = Cast<UStaticMesh>(AssetData.GetAsset());
		if (Asset)
		{
			FString AssetName = Asset->GetName();
			FString Filename = FPaths::Combine(Path, AssetName + Ext);

			UAssetExportTask* ExportTask = NewObject<UAssetExportTask>();
			ExportTask->Object = Asset;
			ExportTask->Filename = Filename;
			ExportTask->bSelected = false;
			ExportTask->bReplaceIdentical = true;
			ExportTask->bAutomated = !bPrompt;
			ExportTask->bPrompt = bPrompt;

			UFbxExportOption* FbxOptions = NewObject<UFbxExportOption>();
			FbxOptions->bASCII = false;
			
			ExportTask->Options = FbxOptions;

			const bool bSuccess = UExporter::RunAssetExportTask(ExportTask);

			if (bSuccess) { UE_LOG(LogTemp, Log, _VACT_FILES_EDITOR_EXPORT_MSG_SUCCESS, *AssetName, *Filename); }
			else { UE_LOG(LogTemp, Error, _VACT_FILES_EDITOR_EXPORT_MSG_FAILED, *AssetName); }

			bPrompt = Options == ECmdVActExportOptions::PromptAll;
		}
	}

}

void FVActFilesEditor::ExportTextures(TArray<FAssetData>& Assets, const FString& Path, ECmdVActExportOptions Options)
{
	FString Ext = "";

	bool bPrompt = false
		|| Options == ECmdVActExportOptions::PromptAll
		|| Options == ECmdVActExportOptions::PromptOnce;

	for (const FAssetData& AssetData : Assets)
	{
		UObject* Asset = Cast<UTexture>(AssetData.GetAsset());
		if (Asset)
		{
			Ext = Asset->IsA<UTextureCube>() ? TEXT(".hdr") : TEXT(".png");
			
			FString AssetName = Asset->GetName();
			FString Filename = FPaths::Combine(Path, AssetName + Ext); ;

			UAssetExportTask* ExportTask = NewObject<UAssetExportTask>();
			ExportTask->Object = Asset;
			ExportTask->Filename = Filename;
			ExportTask->bSelected = false;
			ExportTask->bReplaceIdentical = true;
			ExportTask->bAutomated = !bPrompt;
			ExportTask->bPrompt = bPrompt;

			const bool bSuccess = UExporter::RunAssetExportTask(ExportTask);

			if (bSuccess) { UE_LOG(LogTemp, Log, _VACT_FILES_EDITOR_EXPORT_MSG_SUCCESS, *AssetName, *Filename); }
			else { UE_LOG(LogTemp, Error, _VACT_FILES_EDITOR_EXPORT_MSG_FAILED, *AssetName); }

			bPrompt = Options == ECmdVActExportOptions::PromptAll;
		}
	}

}

void FVActFilesEditor::Cmd_ExportSelectedStaticMeshes(const TArray<FString>& Args)
{
	TArray<FAssetData> SelectedAssets;
	switch (Args.Num())
	{
	case 1:
	{
		_ResolveSelected(SelectedAssets);
		ExportStaticMeshes(SelectedAssets, Args[0]);
		break;
	}
	case 2: 
	{
		_ResolveSelected(SelectedAssets);
		const ECmdVActExportOptions* Option = ExportOptionNameToEnum.Find(Args[1]);
		if (Option) { ExportStaticMeshes(SelectedAssets, Args[0], *Option); }
		break;
	}
	case 0: ;
	default: { UE_LOG(LogTemp, Error, _VACT_FILES_EDITOR_CMD_MSG_FAILED); }
	}
}

void FVActFilesEditor::Cmd_ExportSelectedTextures(const TArray<FString>& Args)
{
	TArray<FAssetData> SelectedAssets;
	switch (Args.Num())
	{
	case 1:
	{
		_ResolveSelected(SelectedAssets);
		ExportTextures(SelectedAssets, Args[0]);
		break;
	}
	case 2:
	{
		_ResolveSelected(SelectedAssets);
		const ECmdVActExportOptions* Option = ExportOptionNameToEnum.Find(Args[1]);
		if (Option) { ExportTextures(SelectedAssets, Args[0], *Option); }
		break;
	}
	case 0:;
	default: { UE_LOG(LogTemp, Error, _VACT_FILES_EDITOR_CMD_MSG_FAILED); }
	}
}

void FVActFilesEditor::_ResolveSelected(TArray<FAssetData>& Assets)
{
	FContentBrowserModule& ContentBrowserModule = FModuleManager::LoadModuleChecked< FContentBrowserModule>("ContentBrowser");
	IContentBrowserSingleton& ContentBrowserSingleton = ContentBrowserModule.Get();
	ContentBrowserSingleton.GetSelectedAssets(Assets);
}