// Copyright Vivaicty Interactive, Inc. All Rights Reserved.

#include "VActOICEditorModule.h"
#include "PropertyEditorModule.h"
#include "OICProfileCustomization.h"
#include "OICProfile.h"
#include "VActOICEditor.h"

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
}

void FVActOICEditorModule::ShutdownModule()
{
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
    PropertyModule.UnregisterCustomClassLayout(UOICProfile::StaticClass()->GetFName());
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActOICEditorModule, VActOICEditor)