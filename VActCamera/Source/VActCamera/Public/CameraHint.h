// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#if WITH_EDITORONLY_DATA
#include "Components/ArrowComponent.h"
#endif

#include "VActCamera.h"
#include "CoreMinimal.h"
#include "Engine/NavigationObjectBase.h"
#include "CameraHint.generated.h"

UCLASS()
class VACTCAMERA_API ACameraHint : public ANavigationObjectBase
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CameraHint")
	FCameraVector Vector;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CameraHint")
	FName CameraHintTag;

public:
	ACameraHint(const FObjectInitializer& ObjectInitializer);

	virtual void BeginPlay() override;

#if WITH_EDITORONLY_DATA
private:
	UPROPERTY()
	TObjectPtr<class UArrowComponent> ArrowComponent;

public:

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CameraHint")
	bool bValidate;

	virtual void Validate() override;

	class UArrowComponent* GetArrowComponent() const;

	virtual void PostEditMove(bool bFinished) override;
#endif
};
