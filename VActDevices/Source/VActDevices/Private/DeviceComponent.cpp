#include "DeviceComponent.h"

#include "EngineUtils.h"
#include "SerialManagerComponent.h"

UDeviceComponent::UDeviceComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

ASerialManager * UDeviceComponent::FindSerialManager()
{
	UWorld* World = GetWorld();
	for (TActorIterator<ASerialManager> It(World); It; ++It)
	{
		return *It;
	}
	return nullptr;
}

void UDeviceComponent::OnRegister()
{
	Super::OnRegister();

	SerialManager = FindSerialManager();
	
}

void UDeviceComponent::OnDeviceInput_Implementation(UDeviceProtocol* Protocol, float DeltaTime)
{

}

void UDeviceComponent::OnDeviceOutput_Implementation(UDeviceProtocol* Protocol, float DeltaTime)
{

}

void UDeviceComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	const bool bManager = SerialManager != nullptr;
	if (bManager)
	{
		USerialManagerComponent* Manager = SerialManager->GetComponent();

		FCOMDevice* InEntry = Manager->Ports.Find(PortIn);
		if (InEntry)
		{
			const bool bProcess = InEntry->bUse && InEntry->bActive && InEntry->InputProtocol;
			if (bProcess) { OnDeviceInput(InEntry->InputProtocol, DeltaTime); }
		}

		FCOMDevice* OutEntry = Manager->Ports.Find(PortOut);
		if (OutEntry)
		{
			const bool bProcess = OutEntry->bUse && OutEntry->bActive && OutEntry->OutputProtocol;
			if (bProcess) { OnDeviceOutput(OutEntry->OutputProtocol, DeltaTime); }
		}
	}
	// ...
}

