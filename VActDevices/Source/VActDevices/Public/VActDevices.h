#pragma once

#if PLATFORM_WINDOWS
#define _CHECK_HANDLE(ExpressionIn)											\
{																			\
	HANDLE Status = (ExpressionIn);											\
	DWORD ErrorCode = GetLastError();										\
	bool bFail = Status == INVALID_HANDLE_VALUE								\
		|| ErrorCode == ERROR_FILE_NOT_FOUND;								\
	if (bFail)																\
	{																		\
		const int32 _ErrorNum = 1024;										\
		char _ErrorMsg[_ErrorNum];											\
		FormatMessageA(														\
			FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,		\
			NULL,															\
			ErrorCode,														\
			MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),						\
			_ErrorMsg,														\
			_ErrorNum,														\
			NULL															\
		);																	\
		const FString ErrorMsg(_ErrorMsg);									\
		UE_LOG(LogTemp, Warning, TEXT("Handle Failed \"%s\""), *ErrorMsg);	\
		return;																\
	}																		\
}

#define _CHECK_COM(ExpressionIn)											\
{																			\
	bool bFail = !(ExpressionIn);											\
	if (bFail)																\
	{																		\
		DWORD ErrorCode = GetLastError();									\
		const int32 _ErrorNum = 1024;										\
		char _ErrorMsg[_ErrorNum];											\
		FormatMessageA(														\
			FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,		\
			NULL,															\
			ErrorCode,														\
			MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),						\
			_ErrorMsg,														\
			_ErrorNum,														\
			NULL															\
		);																	\
		const FString ErrorMsg(_ErrorMsg);									\
		UE_LOG(LogTemp, Warning, TEXT("COM Failed \"%s\""), *ErrorMsg);		\
		return;																\
	}																		\
}

#define _CHECK_SOCKET(ExpressionIn)											\
{																			\
	bool bFail = !(ExpressionIn);											\
	if (bFail)																\
	{																		\
		DWORD ErrorCode = GetLastError();									\
		const int32 _ErrorNum = 1024;										\
		char _ErrorMsg[_ErrorNum];											\
		FormatMessageA(														\
			FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,		\
			NULL,															\
			ErrorCode,														\
			MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),						\
			_ErrorMsg,														\
			_ErrorNum,														\
			NULL															\
		);																	\
		const FString ErrorMsg(_ErrorMsg);									\
		UE_LOG(LogTemp, Warning, TEXT("Socket Failed \"%s\""), *ErrorMsg);		\
		return;																\
	}																		\
}
#define _VACT_DEVICES_PORT_UNKONWN_FORMAT TEXT("_UNKCOM%d")
#elif PLATFORM_MAC

#elif PLATFORM_LINUX
#define _VACT_DEVICES_PORT_UNKONWN_FORMAT TEXT("_unktty%d")
#endif

#define _CHECK_FILE(ExpressionIn)											\
{																			\
	bool bFail = !(ExpressionIn);											\
	if (bFail)																\
	{																		\
		const FString ErrorMsg(TEXT("File Failed"));						\
		UE_LOG(LogTemp, Warning, TEXT("%s"), *ErrorMsg);					\
		return;																\
	}																		\
}

#define _VACT_SUPPORTED_PORT_COUNT 32

#include "VActDevicesTypes.h"

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "VActDevices.generated.h"

class FVActDevicesModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};


USTRUCT()
struct VACTDEVICES_API FVActDevices
{
	GENERATED_BODY()

	static const FString PortNames[];

	//static const TCHAR PortUnkonwnFormat[];

#if WITH_EDITORONLY_DATA
	static void _Debug_VActDevices_Test();
#endif;

	static void _Unsafe_Create(FDevice& Into, FString& ComPort, int32 BaudRate = 19200, FDilim Dilim = FDilim::Space());

	static FDevice Create(FString& ComPort, int32 BaudRate = 19200, FDilim Dilim = FDilim::Space());

	static void _Unsafe_Write(FDevice& Into, FString Data);

	static void Write(FDevice& Device, FString Data);

	static void _Unsafe_Read(FString& Into, FDevice& Device, int32 TimeOut = 1);

	static FString Read(FDevice& Device, int32 TimeOut = 1);

	static void _Unsafe_Destory(FDevice& Device, bool bData = false);

	static void Destory(FDevice& Device, bool bData = false);

	static FORCEINLINE char* FStringToChar(FString String)
	{
		std::wstring Filter = *String;
		Filter.push_back('\0');
		return (char*)Filter.c_str();
	}

	static void _Unsafe_Scan(TArray<FDeviceInfo>& Into, EDeviceType Types);

	static void Scan(TArray<FDeviceInfo>& Into, EDeviceType Types = EDeviceType::Bluetooth);

	static void _Unsafe_Connect(bool& Into, FConnect& Socket, FDeviceInfo& Info);

	static bool Connect(FConnect& Socket, FDeviceInfo& Info);

	//static FString ReadString(FDevice& Device, FChar Dilim);

	//static int32 ReadInt(FDevice& Device, FChar Dilim);

	//static float ReadFloat(FDevice& Device, FChar Dilim);

	//static FVector ReadVector(FDevice& Device, FChar Dilim);

	//static FQuat ReadQuat(FDevice& Device, FChar Dilim);

	//static FRotator ReadRotator(FDevice& Device, FChar Dilim);

};