#pragma once

#include "IAssetTools.h"
#include "Modules/ModuleManager.h"
#include "AssetToolsModule.h"
#include "ContentBrowserModule.h"
#include "AssetTypeActions_Base.h"


template<typename T, typename TypeAssetAction>
class UOICProfileFactory : public UFactory
{
	UOICProfileFactory()
	{
		bCreateNew = true;
		bEditAfterNew = true;
		SupportedClass = T::StaticClass();
	}

	virtual UObject* FactoryCreateNew(UClass* Class, UObject* InParent, FName Name, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn) override
	{
		T* NewAsset = NewObject<T>(InParent, Class, Name, Flags | RF_Transactional);
		return NewAsset;
	}
};



class UOICProfile;

template<typename T, typename TypeAssetAction>
class TDefaultAssetTypeActions : public FAssetTypeActions_Base
{
public:
	FOICProfileAssetTypeActions(EAssetTypeCategories::Type InAssetCategory)
		: MyAssetCategory(InAssetCategory)
	{
		
	}

	virtual FText GetName() const override
	{
		return LOCTEXT("F" LOCTEXT_ASSETTYPENAME "AssetTypeActionsName", LOCTEXT_ASSETTYPENAME);
	}
	virtual FColor GetTypeColor() const override
	{
		return FColor::Cyan;
	}

	virtual UClass* GetSupportedClass() const override
	{
		return T::StaticClass();
	}

	virtual void GetActions(const TArray<UObject*>& InObjects, struct FToolMenuSection& Section) override
	{
		auto Profiles = GetTypedWeakObjectPtrs<T>(InObjects);

		Section.AddMenuEntry(
			TypeAssetAction::AssetTypeName "_Create" TypeAssetAction::AssetTypeName,
			LOCTEXT(TypeAssetAction::AssetTypeName "_Create" TypeAssetAction::AssetTypeName, TypeAssetAction::AssetTypeNameView " Profile"),
			LOCTEXT(TypeAssetAction::AssetTypeName "_Create" TypeAssetAction::AssetTypeName "Tooltip", "Creates new " TypeAssetAction::AssetTypeNameView "."),
			FSlateIcon(),
			FUIAction(
				FExecuteAction::CreateSP(this, &FOICProfileAssetTypeActions::ExecuteCreate, Profiles[0]),
				FCanExecuteAction()
			)
		);
	}

	virtual uint32 GetCategories() override;

private:
	EAssetTypeCategories::Type MyAssetCategory;

	void ExecuteCreate(TWeakObjectPtr<T> ObjectPtr);
};