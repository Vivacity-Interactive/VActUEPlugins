// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "SerialManagerComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/Info.h"
#include "SerialManager.generated.h"

class USerialManagerComponent;

UCLASS()
class VACTDEVICES_API ASerialManager : public AInfo
{
	GENERATED_BODY()

private:
	UPROPERTY(Category = SerialManager, VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	TObjectPtr<USerialManagerComponent> Component;
	
public:

	ASerialManager();

	USerialManagerComponent* GetComponent() const { return Component; }
};
