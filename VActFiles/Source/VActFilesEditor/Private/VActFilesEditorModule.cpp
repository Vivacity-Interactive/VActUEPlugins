// Copyright Vivaicty Interactive, Inc. All Rights Reserved.

#include "VActFilesEditorModule.h"
#include "VActFilesEditor.h"

#define LOCTEXT_NAMESPACE "FVActFilesEditorModule"

void FVActFilesEditorModule::StartupModule()
{
    IConsoleManager& ConsoleManager = IConsoleManager::Get();
    
    ConsoleManager.RegisterConsoleCommand(
        TEXT("VActExportSelectedMeshes"),
        TEXT("Exports selected static meshes to FBX, Usage: VActExportSelectedMeshes <Path> [PromptAll|NoPrompt|PromptOnce]"),
        FConsoleCommandWithArgsDelegate::CreateStatic(&FVActFilesEditor::Cmd_ExportSelectedStaticMeshes),
        ECVF_Default
    );
    
    ConsoleManager.RegisterConsoleCommand(
        TEXT("VActExportSelectedTextures"),
        TEXT("Exports selected textures to PNG, Usage: VActExportSelectedTextures <Path> [PromptAll|NoPrompt|PromptOnce]"),
        FConsoleCommandWithArgsDelegate::CreateStatic(&FVActFilesEditor::Cmd_ExportSelectedTextures),
        ECVF_Default
    );
}

void FVActFilesEditorModule::ShutdownModule()
{
	
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActFilesEditorModule, VActFilesEditor)
