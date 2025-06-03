#include "DeviceProtocol.h"

void UDeviceProtocol::Open_Implementation(FDevice& Device, float DeltaTime)
{

}

void UDeviceProtocol::Close_Implementation(FDevice& Device, float DeltaTime)
{

}

void UDeviceProtocol::Read_Implementation(FDevice& Device, float DeltaTime)
{
	InBuffer = FVActDevices::Read(Device, 1);
}


void UDeviceProtocol::Write_Implementation(FDevice& Device, float DeltaTime)
{
	FVActDevices::Write(Device, OutBuffer);
}