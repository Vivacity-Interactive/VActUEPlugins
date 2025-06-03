// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/NavigationObjectBase.h"
#include "Empty.generated.h"

UCLASS()
class VACTBASE_API AEmpty : public ANavigationObjectBase
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Empty)
	FName EmptyTag;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Empty)
	bool bValidate;

public:
	AEmpty(const FObjectInitializer& ObjectInitializer);

	virtual void Validate() override;

#if WITH_EDITORONLY_DATA
private:
	UPROPERTY()
	TObjectPtr<class UArrowComponent> ArrowComponent;
public:
#endif

#if WITH_EDITORONLY_DATA
	class UArrowComponent* GetArrowComponent() const;
#endif

};
