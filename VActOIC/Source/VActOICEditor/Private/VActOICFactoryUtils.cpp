#include "VActOICFactoryUtils.h"

#include "OICProfile.h"

UOICProfileFactory::UOICProfileFactory()
{
	bCreateNew = true;
	bEditAfterNew = true;
	SupportedClass = UOICProfile::StaticClass();
}

UObject* UOICProfileFactory::FactoryCreateNew(UClass* Class, UObject* InParent, FName Name, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn)
{
	UOICProfile* NewOICProfile = NewObject<UOICProfile>(InParent, Class, Name, Flags | RF_Transactional);
	return NewOICProfile;
}