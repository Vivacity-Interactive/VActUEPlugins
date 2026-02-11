#pragma once

#include "AssetTypeActions_Base.h"

class UOICProfile;

class FOICProfileAssetTypeActions : public FAssetTypeActions_Base
{
public:
	FOICProfileAssetTypeActions(EAssetTypeCategories::Type InAssetCategory);

	virtual FText GetName() const override;
	virtual FColor GetTypeColor() const override;
	virtual UClass* GetSupportedClass() const override;
	virtual void GetActions(const TArray<UObject*>& InObjects, struct FToolMenuSection& Section) override;
	virtual uint32 GetCategories() override;

private:
	EAssetTypeCategories::Type MyAssetCategory;

	void ExecuteCreate(TWeakObjectPtr<UOICProfile> ObjectPtr);
};
