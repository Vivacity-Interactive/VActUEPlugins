// Copyright Vivaicty Interactive, Inc. All Rights Reserved.

#include "VActOICEditorModule.h"
#include "PropertyEditorModule.h"
#include "OICProfileCustomization.h"
#include "OICProfile.h"

#define LOCTEXT_NAMESPACE "FVActOICEditorModule"

void FVActOICEditorModule::StartupModule()
{
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
    PropertyModule.RegisterCustomClassLayout(UOICProfile::StaticClass()->GetFName(), FOnGetDetailCustomizationInstance::CreateStatic(&FOICProfileCustomization::MakeInstance));
}

void FVActOICEditorModule::ShutdownModule()
{
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
    PropertyModule.UnregisterCustomClassLayout(UOICProfile::StaticClass()->GetFName());
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActOICEditorModule, VActOICEditor)