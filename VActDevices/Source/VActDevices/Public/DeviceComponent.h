// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "SerialManager.h"

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "DeviceComponent.generated.h"

class ASerialManager;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class VACTDEVICES_API UDeviceComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0", ClampMax = "31"))
	int32 PortOut;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0", ClampMax = "31"))
	int32 PortIn;

	TWeakObjectPtr<ASerialManager> SerialManager;

public:	
	UDeviceComponent();

	ASerialManager* FindSerialManager();

protected:
	virtual void OnRegister() override;

public:	
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void OnDeviceInput(UDeviceProtocol* Protocol, float DeltaTime);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void OnDeviceOutput(UDeviceProtocol* Protocol, float DeltaTime);

		
};
