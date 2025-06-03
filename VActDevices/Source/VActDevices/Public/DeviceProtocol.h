// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "VActDevices.h"
#include "VActDevicesTypes.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "DeviceProtocol.generated.h"

struct FDevice;

UCLASS()
class VACTDEVICES_API UDeviceProtocol : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString InBuffer;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString OutBuffer;
	
public:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Open(FDevice& Device, float DeltaTime);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Close(FDevice& Device, float DeltaTime);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Read(FDevice& Device, float DeltaTime);

	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Write(FDevice& Device, float DeltaTime);

};
