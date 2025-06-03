// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "VActDevices.h"
#include "VActDevicesTypes.h"
#include "DeviceProtocol.h"

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "SerialManagerComponent.generated.h"

class UDeviceProtocol;

UENUM(BlueprintType)
enum class ECOMIO : uint8
{
	None = 0,
	Input = 1 << 0,
	Output = 1 << 1,
	IO = Input | Output
};

// UENUM(BlueprintType)
// enum class ECOMBaudRate : uint8
// {
// 	None = 0,
// 	BR_110 = 110,
// 	BR_300 = 300,
// 	BR_600 = 600,
// 	BR_1200 = 1200,
// 	BR_2400 = 2400,
// 	BR_4800 = 4000,
// 	BR_9600 = 9600,
// 	BR_19200 = 19200,
// 	BR_38400 = 38400,
// 	BR_57600 = 57600,
// 	BR_115200 = 115200,
// 	BR_128000 = 128000,
// 	BR_256000 = 256000,
// 	BR_500000 = 500000,
// 	BR_921600 = 921600,
// 	BR_1000000 = 1000000,
// 	BR_1500000 = 1500000,
// 	BR_2000000 = 2000000,
// 	BR_3000000 = 3000000
// };

UENUM(BlueprintType)
enum class ECOMModes : uint8
{
	None = 0,
	M_8N1,
	M_8O1,
	M_8E1,
	M_8N2,
	M_8O2,
	M_8E2,
	M_7N1,
	M_7O1,
	M_7E1,
	M_7N2,
	M_7O2,
	M_7E2,
	M_6N1,
	M_6O1,
	M_6E1,
	M_6N2,
	M_6O2,
	M_6E2,
	M_5N1,
	M_5O1,
	M_5E1,
	M_5N2,
	M_5O2,
	M_5E2
};

USTRUCT(BlueprintType)
struct FCOMDevice
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString PortName;

	//UPROPERTY(EditAnywhere, BlueprintReadWrite)
	//int32 PortId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool bUse;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Rate;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	ECOMModes Modes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	ECOMIO IO;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	bool bActive;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TSubclassOf<UDeviceProtocol> InputProtocolClass;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TSubclassOf<UDeviceProtocol> OutputProtocolClass;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TObjectPtr<UDeviceProtocol> InputProtocol;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TObjectPtr<UDeviceProtocol> OutputProtocol;

	FDevice Device;

public:
	static const FCOMDevice DefaultCOMDevice;

	static FCOMDevice Default();

	~FCOMDevice();
};

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class VACTDEVICES_API USerialManagerComponent : public USceneComponent
{
	GENERATED_BODY()

	//TMap<int32, int32> PortMap;

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TMap<int32, FCOMDevice> Ports;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TArray<FDeviceInfo> Devices;

	//UPROPERTY(EditANywhere, BlueprintReadWrite)
	//TArray<FCOMDevice> Ports;

public:
	USerialManagerComponent();

	UFUNCTION(CallInEditor)
	void ScanDevices();

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:	
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

		
};
