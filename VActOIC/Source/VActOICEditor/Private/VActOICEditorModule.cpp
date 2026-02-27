// Copyright Vivaicty Interactive, Inc. All Rights Reserved.

#include "VActOICEditorModule.h"
#include "PropertyEditorModule.h"
#include "OICProfileCustomization.h"
#include "OICProfile.h"
#include "VActOICEditor.h"

#include "VActOICAssetTypeActionsUtils.h"

#include "AssetToolsModule.h"
#include "ContentBrowserModule.h"

#define LOCTEXT_NAMESPACE "FVActOICEditorModule"

void FVActOICEditorModule::StartupModule()
{
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
    PropertyModule.RegisterCustomClassLayout(UOICProfile::StaticClass()->GetFName(), FOnGetDetailCustomizationInstance::CreateStatic(&FOICProfileCustomization::MakeInstance));

    IConsoleManager& ConsoleManager = IConsoleManager::Get();

    ConsoleManager.RegisterConsoleCommand(
        TEXT("VActExportToOIC"),
        TEXT("Exports selected levels or active level to OIC, Usage: VActExportToOIC <Path> [PerLevel|PerWorld|AllCombined]"),
        FConsoleCommandWithArgsDelegate::CreateStatic(&FVActOICEditor::Cmd_ExportToOICAsset),
        ECVF_Default
    );

    IAssetTools& AssetTools = FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();
    VActAssetCategoryBit = AssetTools.RegisterAdvancedAssetCategory(FName(TEXT("VAct")), LOCTEXT("VActAssetCategory", "VAct"));
    RegisterAssetTypeAction(AssetTools, MakeShareable(new FOICProfileAssetTypeActions(VActAssetCategoryBit)));
}

void FVActOICEditorModule::ShutdownModule()
{
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
    PropertyModule.UnregisterCustomClassLayout(UOICProfile::StaticClass()->GetFName());

    if (FModuleManager::Get().IsModuleLoaded("AssetTools"))
    {
        IAssetTools& AssetTools = FModuleManager::GetModuleChecked<FAssetToolsModule>("AssetTools").Get();
        for (int32 Index = 0; Index < CreatedAssetTypeActions.Num(); ++Index)
        {
            AssetTools.UnregisterAssetTypeActions(CreatedAssetTypeActions[Index].ToSharedRef());
        }
    }
    CreatedAssetTypeActions.Empty();

}

void FVActOICEditorModule::RegisterAssetTypeAction(IAssetTools& AssetTools, TSharedRef<IAssetTypeActions> Action)
{
    AssetTools.RegisterAssetTypeActions(Action);
    CreatedAssetTypeActions.Add(Action);
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActOICEditorModule, VActOICEditor)