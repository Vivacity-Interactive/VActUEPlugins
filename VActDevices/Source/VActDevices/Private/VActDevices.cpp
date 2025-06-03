#include "VActDevices.h"

#if PLATFORM_WINDOWS
const FString FVActDevices::PortNames[_VACT_SUPPORTED_PORT_COUNT+1] = {
	TEXT("_NONE"),
	TEXT("COM1"),
	TEXT("COM2"),
	TEXT("COM3"),
	TEXT("COM4"),
	TEXT("COM5"),
	TEXT("COM6"),
	TEXT("COM7"),
	TEXT("COM8"),
	TEXT("COM9"),
	TEXT("COM10"),
	TEXT("COM11"),
	TEXT("COM12"),
	TEXT("COM13"),
	TEXT("COM14"),
	TEXT("COM15"),
	TEXT("COM16"),
	TEXT("COM17"),
	TEXT("COM18"),
	TEXT("COM19"),
	TEXT("COM20"),
	TEXT("COM21"),
	TEXT("COM22"),
	TEXT("COM23"),
	TEXT("COM24"),
	TEXT("COM25"),
	TEXT("COM26"),
	TEXT("COM27"),
	TEXT("COM28"),
	TEXT("COM29"),
	TEXT("COM30"),
	TEXT("COM31"),
	TEXT("COM32")
};
const TCHAR FVActDevices::PortUnkonwnFormat[] = TEXT("_UNKCOM%d");

//#define BTHPROTO_RFCOMM  0x0003
//#define BTHPROTO_L2CAP   0x0100
//#define BTHPROTO_SCO      0x0001

#elif defined(PLATFORM_MAC) || defined(PLATFORM_LINUX)
const FString FVActDevices::PortNames[_VACT_SUPPORTED_PORT_COUNT + 1] = {
	TEXT("_none"),
	TEXT("ttyS0"),
	TEXT("ttyS1"),
	TEXT("ttyS2"),
	TEXT("ttyS3"),
	TEXT("ttyS4"),
	TEXT("ttyS5"),
	TEXT("ttyS6"),
	TEXT("ttyS7"),
	TEXT("ttyS8"),
	TEXT("ttyS9"),
	TEXT("ttyS10"),
	TEXT("ttyS11"),
	TEXT("ttyS12"),
	TEXT("ttyS13"),
	TEXT("ttyS14"),
	TEXT("ttyS15"),
	TEXT("ttyUSB0"),
	TEXT("ttyUSB1"),
	TEXT("ttyUSB2"),
	TEXT("ttyUSB3"),
	TEXT("ttyUSB4"),
	TEXT("ttyUSB5"),
	TEXT("ttyAMA0"),
	TEXT("ttyAMA1"),
	TEXT("ttyACM0"),
	TEXT("ttyACM1"),
	TEXT("rfcomm0"),
	TEXT("rfcomm1"),
	TEXT("ircomm0"),
	TEXT("ircomm1"),
	TEXT("cuau0"),
	TEXT("cuau1")
};
const TCHAR FVActDevices::PortUnkonwnFormat[] = TEXT("_unktty%d");
#endif

#define LOCTEXT_NAMESPACE "FVActDevicesModule"

void FVActDevicesModule::StartupModule()
{

#if WITH_EDITORONLY_DATA
	//FVActDevices::_Debug_VActDevices_Test();
#endif;
}

void FVActDevicesModule::ShutdownModule()
{
	
	
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActDevicesModule, VActDevices)


#if WITH_EDITORONLY_DATA

void FVActDevices::_Debug_VActDevices_Test()
{
	//FDevice Device = FVActDevices::Create("\\\\.\\COM5", 9600, FDilim::Line());
	FString ComPort = TEXT("COM5");
	UE_LOG(LogTemp, Display, TEXT("VActDevices Test Opening \"%s\""), *ComPort);

	FDevice Device = FVActDevices::Create(ComPort, 9600, FDilim::Line());
	
	FString Result = FVActDevices::Read(Device, 1);
	UE_LOG(LogTemp, Display, TEXT("VActDevices Test Read %s"), *Result);
	
	FVActDevices::Destory(Device);
	UE_LOG(LogTemp, Display, TEXT("VActDevices Test Success"));
}

#endif;

void FVActDevices::_Unsafe_Create(FDevice& Into, FString& ComPort, int32 BaudRate, FDilim Dilim)
{
	Into.bConnected = false;

	_CHECK_HANDLE(Into.Handle = CreateFile(*ComPort, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL));
	
	Into.Desc = { 0 };
	SecureZeroMemory(&Into.Desc, sizeof(DCB));
	Into.Desc.DCBlength = sizeof(DCB);
	_CHECK_COM(GetCommState(Into.Handle, &Into.Desc));
	
	Into.Dilim = Dilim;
	Into.Desc.BaudRate = (_VACT_DWORD) BaudRate;
	Into.Desc.ByteSize = 8;
	Into.Desc.StopBits = ONESTOPBIT;
	Into.Desc.Parity = NOPARITY;
	Into.Desc.fDtrControl = DTR_CONTROL_ENABLE;

	_CHECK_COM(SetCommState(Into.Handle, &Into.Desc));
	
	// Into.TimeOut = { 0 };
	// Into.TimeOut.ReadIntervalTimeout = 10;
	// Into.TimeOut.ReadTotalTimeoutConstant = 50;
	// Into.TimeOut.ReadTotalTimeoutMultiplier = 10;
	// Into.TimeOut.WriteTotalTimeoutConstant = 10;
	// Into.TimeOut.WriteTotalTimeoutMultiplier = 10;

	// _CHECK_COM(SetCommTimeouts(Into.Handle, &Into.TimeOut));
	
	_CHECK_COM(PurgeComm(Into.Handle, PURGE_RXCLEAR | PURGE_TXCLEAR));

	Into.bConnected = true;
}

FDevice FVActDevices::Create(FString& ComPort, int32 BaudRate, FDilim Dilim)
{
	FDevice Device;
	_Unsafe_Create(Device, ComPort, BaudRate, Dilim);
	return Device;
}


void FVActDevices::_Unsafe_Write(FDevice& Into, FString Data)
{
	_VACT_DWORD Count;
	_VACT_DWORD Len = Data.Len();
	_CHECK_FILE(WriteFile(Into.Handle, (void*)FStringToChar(Data), Len, &Count, NULL));
	_CHECK_COM(ClearCommError(Into.Handle, &Into.Error, &Into.Status));
}

void FVActDevices::Write(FDevice& Device, FString Data)
{
	_Unsafe_Write(Device, Data);
}

void FVActDevices::_Unsafe_Read(FString& Into, FDevice& Device, int32 TimeOut)
{
	_VACT_DWORD Count;
	_VACT_CHAR Buffer[_VACT_BUFFER_SIZE + 1] = { 0 };
	bool bReading = false;

	_VACT_SIZE_T Start = time(nullptr);
	
	bool _bSuccess = false;

	_CHECK_COM(ClearCommError(Device.Handle, &Device.Error, &Device.Status));

	while ((time(nullptr) - Start) < TimeOut)
	{
		_CHECK_FILE(ReadFile(Device.Handle, Buffer, _VACT_BUFFER_SIZE, &Count, NULL));
		bool bStart = ((int32)Device.Dilim.Flag & (int32)EDilimFlag::Open) || bReading || Buffer[0] == Device.Dilim.Open;
		if (bStart)
		{
			bReading = true;
			bool bEnd = Buffer[0] == Device.Dilim.Close;
			// needs escape implemented
			if (bEnd) { return; }
			Into.Append(Buffer, _VACT_BUFFER_SIZE);
		}
	}
}

//static bool TokenContext(const TCHAR& Char, const TCHAR& Escape, FVActParseCursor& Cursor)
//{
//	bool _bNext = Cursor.IsValid() && Char == *Cursor && ++Cursor, bValid = false, bEscape = false;
//	const TCHAR* _To = Cursor.To;
//
//	while (_bNext && !bValid)
//	{
//		const TCHAR& _Char = *Cursor;
//		bValid = !bEscape && Char == _Char;
//		bEscape = ((int32)Device.Dilim.Flag & (int32)EDilimFlag::Escape) || _Char == Device.Dilim.Escape && !bEscape;
//		++Cursor; // It may incement once to much
//	}
//
//	if (!bValid) { Cursor.To = _To; }
//	return bValid;
//}

FString FVActDevices::Read(FDevice& Device, int32 TimeOut)
{
	FString Data;
	_Unsafe_Read(Data, Device, TimeOut);
	return Data;
}

void FVActDevices::_Unsafe_Destory(FDevice& Device, bool bData)
{
	if (Device.bConnected)
	{
		Device.bConnected = false;
		_CHECK_COM(CloseHandle(Device.Handle));
	}
}

void FVActDevices::Destory(FDevice& Device, bool bData)
{
	_Unsafe_Destory(Device);
}

void FVActDevices::_Unsafe_Scan(TArray<FDeviceInfo>& Into, EDeviceType Types)
{
	BLUETOOTH_DEVICE_SEARCH_PARAMS SearchParams;
	SecureZeroMemory(&SearchParams, sizeof(BLUETOOTH_DEVICE_SEARCH_PARAMS));
	SearchParams.dwSize = sizeof(BLUETOOTH_DEVICE_SEARCH_PARAMS);
	SearchParams.fReturnAuthenticated = TRUE;
	SearchParams.fReturnRemembered = TRUE;
	SearchParams.fReturnUnknown = TRUE;
	SearchParams.fReturnConnected = TRUE;
	SearchParams.fIssueInquiry = TRUE;
	SearchParams.cTimeoutMultiplier = 5;

	FDeviceInfo _Info;
	SecureZeroMemory(&_Info.Info, sizeof(_Info.Info));
	_Info.Info.dwSize = sizeof(_Info.Info);
	
	HANDLE HanldeFind = BluetoothFindFirstDevice(&SearchParams, &_Info.Info);
	BOOL bNext = HanldeFind != NULL;
	while (bNext)
	{
		_Info.Type = EDeviceType::Bluetooth;
		_Info.DeviceName = _Info.Info.szName;
		_Info.bConnected = _Info.Info.fConnected;
		_Info.Adress = FString::FromBlob(_Info.Info.Address.rgBytes, sizeof(_Info.Info.Address.rgBytes));
		Into.Add(_Info);
		bNext = BluetoothFindNextDevice(HanldeFind, &_Info.Info);
	}

	BluetoothFindDeviceClose(HanldeFind);
}

void FVActDevices::Scan(TArray<FDeviceInfo>& Into, EDeviceType Types)
{
	_Unsafe_Scan(Into, Types);
}

void FVActDevices::_Unsafe_Connect(bool& Into, FConnect& Socket, FDeviceInfo& Info)
{
	if (Info.Info.fConnected)
	{
		Socket.Socket = socket(AF_BTH, SOCK_STREAM, BTHPROTO_RFCOMM);
		if (Socket.Socket != INVALID_SOCKET)
		{
			_SOCKADDR_BTH Adress;
			SecureZeroMemory(&Adress, sizeof(_SOCKADDR_BTH));
			Adress.addressFamily = AF_BTH;
			Adress.btAddr = Info.Info.Address.ullLong;
			Adress.serviceClassId = SerialPortServiceClass_UUID;
			Adress.port = BT_PORT_ANY;

			_CHECK_SOCKET(connect(Socket.Socket, (SOCKADDR*)&Adress, sizeof(_SOCKADDR_BTH))); 
			//note potential leek unclosed socket
		}
	}
}

bool FVActDevices::Connect(FConnect& Socket, FDeviceInfo& Info)
{
	bool bSuccess;
	_Unsafe_Connect(bSuccess, Socket, Info);
	return bSuccess;
}