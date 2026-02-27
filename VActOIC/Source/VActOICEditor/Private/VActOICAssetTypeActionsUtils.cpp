#include "VActOICAssetTypeActionsUtils.h"
#include "VActOICFactoryUtils.h"

#include "AssetToolsModule.h"
#include "ContentBrowserModule.h"

#include "OICProfile.h"

#define LOCTEXT_NAMESPACE "VActOICAssetTypeActionsUtils"


#define LOCTEXT_ASSETTYPENAME "OICProfile"
#define LOCTEXT_ASSETTYPENAME_VIEW "OIC Profile"

FOICProfileAssetTypeActions::FOICProfileAssetTypeActions(EAssetTypeCategories::Type InAssetCategory)
	: MyAssetCategory(InAssetCategory)
{}

FText FOICProfileAssetTypeActions::GetName() const
{
	return LOCTEXT("F" LOCTEXT_ASSETTYPENAME "AssetTypeActionsName", LOCTEXT_ASSETTYPENAME);
}

FColor FOICProfileAssetTypeActions::GetTypeColor() const
{
	return FColor::Cyan;
}

UClass* FOICProfileAssetTypeActions::GetSupportedClass() const
{
	return UOICProfile::StaticClass();
}

uint32 FOICProfileAssetTypeActions::GetCategories()
{
	return MyAssetCategory;
}

void FOICProfileAssetTypeActions::GetActions(const TArray<UObject*>& InObjects, FToolMenuSection& Section)
{
	auto Profiles = GetTypedWeakObjectPtrs<UOICProfile>(InObjects);

	Section.AddMenuEntry(
		LOCTEXT_ASSETTYPENAME "_Create" LOCTEXT_ASSETTYPENAME,
		LOCTEXT(LOCTEXT_ASSETTYPENAME "_Create" LOCTEXT_ASSETTYPENAME, LOCTEXT_ASSETTYPENAME_VIEW " Profile"),
		LOCTEXT(LOCTEXT_ASSETTYPENAME "_Create" LOCTEXT_ASSETTYPENAME "Tooltip", "Creates new " LOCTEXT_ASSETTYPENAME_VIEW "."),
		FSlateIcon(),
		FUIAction(
			FExecuteAction::CreateSP(this, &FOICProfileAssetTypeActions::ExecuteCreate, Profiles[0]),
			FCanExecuteAction()
		)
	);
}

void FOICProfileAssetTypeActions::ExecuteCreate(TWeakObjectPtr<UOICProfile> ObjectPtr)
{
	FAssetToolsModule& AssetToolsModule = FModuleManager::GetModuleChecked<FAssetToolsModule>("AssetTools");
	FContentBrowserModule& ContentBrowserModule = FModuleManager::LoadModuleChecked<FContentBrowserModule>("ContentBrowser");
	const FString ObjectSuffix(TEXT("_" LOCTEXT_ASSETTYPENAME));

	if (UOICProfile* Object = ObjectPtr.Get())
	{
		FString EffectiveObjectName = Object->GetName();

		const FString ObjectPathName = Object->GetOutermost()->GetPathName();
		const FString LongPackagePath = FPackageName::GetLongPackagePath(ObjectPathName);
		const FString NewTileMapDefaultPath = LongPackagePath / EffectiveObjectName;

		FString AssetName;
		FString PackageName;
		AssetToolsModule.Get().CreateUniqueAssetName(NewTileMapDefaultPath, ObjectSuffix, PackageName, AssetName);
		const FString PackagePath = FPackageName::GetLongPackagePath(PackageName);
	}
}
#undef LOCTEXT_ASSETTYPENAME
#undef LOCTEXT_ASSETTYPENAME_VIEW

#undef LOCTEXT_NAMESPACE