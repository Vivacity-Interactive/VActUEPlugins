// Fill out your copyright notice in the Description page of Project Settings.


#include "SerialManagerComponent.h"

const FCOMDevice FCOMDevice::DefaultCOMDevice = FCOMDevice::Default();

FCOMDevice FCOMDevice::Default()
{
	FCOMDevice COMDevice;
	COMDevice.PortName = TEXT("");
	COMDevice.bUse = false;
	COMDevice.Rate = 19200;
	COMDevice.Modes = ECOMModes::None;
	COMDevice.IO = ECOMIO::None;
	COMDevice.bActive = false;
	COMDevice.InputProtocolClass = nullptr;
	COMDevice.OutputProtocolClass = nullptr;
	COMDevice.InputProtocol = nullptr;
	COMDevice.OutputProtocol = nullptr;
	COMDevice.Device = { 0 };
	return COMDevice;
}

FCOMDevice::~FCOMDevice()
{
	if (bActive)
	{
		FVActDevices::Destory(Device);
		bActive = false;
	}
}

// Sets default values for this component's properties
USerialManagerComponent::USerialManagerComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

// Called when the game starts
void USerialManagerComponent::BeginPlay()
{
	Super::BeginPlay();
	// Move this to use only
	for (TPair<int32, FCOMDevice>& Pair : Ports)
	{
		FCOMDevice& Entry = Pair.Value;
		
		if (Entry.InputProtocolClass) { Entry.InputProtocol = NewObject<UDeviceProtocol>(this, Entry.InputProtocolClass); }
		if (Entry.OutputProtocolClass) { Entry.OutputProtocol = NewObject<UDeviceProtocol>(this, Entry.OutputProtocolClass); }
	}
}

void USerialManagerComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	if (EndPlayReason != EEndPlayReason::EndPlayInEditor)
	{
		for (TPair<int32, FCOMDevice>&Pair : Ports)
		{
			FCOMDevice& Entry = Pair.Value;
			if (Entry.bActive)
			{
				FVActDevices::Destory(Entry.Device);
				Entry.bActive = false;
			}
		}
	}

}


// Called every frame
void USerialManagerComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	for (TPair<int32, FCOMDevice>& Pair : Ports)
	{
		FCOMDevice& Entry = Pair.Value;
		
		bool bOpen = Entry.bUse && !Entry.bActive;
		if (bOpen) 
		{
			const bool bKnownPort = 0 <= Pair.Key && Pair.Key <= _VACT_SUPPORTED_PORT_COUNT;
			Entry.PortName = bKnownPort ? FVActDevices::PortNames[Pair.Key] : FString::Printf(_VACT_DEVICES_PORT_UNKONWN_FORMAT, Pair.Key);
			UE_LOG(LogTemp, Log, TEXT("Opening %s"), *Entry.PortName);
			Entry.Device = FVActDevices::Create(Entry.PortName, Entry.Rate, FDilim::Line());
			// Also Instantiate the Protocols Here if needed (realtime support)
			if (Entry.InputProtocol) { Entry.InputProtocol->Open(Entry.Device, DeltaTime); }
			if (Entry.OutputProtocol) { Entry.OutputProtocol->Open(Entry.Device, DeltaTime); }
			Entry.bActive = true;
		}
		bool bClose = !Entry.bUse && Entry.bActive;
		if (bClose)
		{
			UE_LOG(LogTemp, Log, TEXT("Closing %s"), *Entry.PortName);
			// Also Instantiate the Protocols Here if needed (realtime support)
			if (Entry.InputProtocol) { Entry.InputProtocol->Close(Entry.Device, DeltaTime); }
			if (Entry.OutputProtocol) { Entry.OutputProtocol->Close(Entry.Device, DeltaTime); }
			FVActDevices::Destory(Entry.Device);
			Entry.bUse = false;
			Entry.bActive = false;
		}
		bool bAct = Entry.bUse && Entry.bActive;
		if (bAct)
		{
			bool bInput = ((int32)Entry.IO & (int32)ECOMIO::Input) && Entry.InputProtocol;
			if (bInput) { Entry.InputProtocol->Read(Entry.Device, DeltaTime); }

			bool bOutput = ((int32)Entry.IO & (int32)ECOMIO::Output) && Entry.OutputProtocol;
			if (bOutput) { Entry.OutputProtocol->Write(Entry.Device, DeltaTime); }
		}
	}
}

void USerialManagerComponent::ScanDevices()
{
	Devices.Empty();
	FVActDevices::Scan(Devices);
}

//class _FScanDevicesAsyc : public FRunnable
//{
//public:
//	virtual uint32 Run() override
//	{
//		AsyncTask(ENamedThreads::GameThread, []()
//        {
//            // Editor-only operation here
//            if (GEditor)
//            {
//                // Example: Accessing a specific editor function
//                GEditor->GetEditorWorldContext().World()->GetWorldSettings()->bEnableWorldComposition = true;
//            }
//        });
//
//		return 0;
//	}
//
//}