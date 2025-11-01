#include "STTModelAsset.h"

#include "VActSTT.h"

USTTModelAsset::USTTModelAsset()
	: Model()
{
}

void USTTModelAsset::BeginDestroy()
{
	Super::BeginDestroy();

#if WITH_EDITOR
	UE_LOG(LogTemp, Warning, TEXT("model destory '%s'"), *GetNameSafe(this));
#endif
	FVActSTT::UnloadModel(Model, true);
}