#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "VActOICFactoryUtils.generated.h"

UCLASS()
class UOICProfileFactory : public UFactory
{
	GENERATED_BODY()

	UOICProfileFactory();

	virtual UObject* FactoryCreateNew(UClass* Class, UObject* InParent, FName Name, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn) override;
};
