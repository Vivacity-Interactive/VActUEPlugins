#pragma once

#include "VActSTTTypes.h"

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "STTModelAsset.generated.h"

struct FSTTModel;

UCLASS(BlueprintType, Blueprintable, EditInlineNew)
class VACTSTT_API USTTModelAsset : public UDataAsset
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, Category = "MLModel")
	FName Name;

	UPROPERTY(EditAnywhere, Category = "MLModel")
	FString Title;

#if WITH_EDITORONLY_DATA
	UPROPERTY(EditAnywhere, Category = "MLModel")
	FString Note;
#endif

	UPROPERTY(EditAnywhere, Category = "MLModel", meta = (FileBasePath = "/Game/MLModels", FilePathFilter = "*.bin"))
	FFilePath FilePath;

	UPROPERTY(EditAnywhere, Category = "MLModel")
	FSTTModel Model;

public:
	USTTModelAsset();

protected:
	virtual void BeginDestroy() override;
};
