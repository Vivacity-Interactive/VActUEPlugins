#pragma once

#if PLATFORM_WINDOWS
#include <Windows.h>
#include <bluetoothapis.h>
#include <bthsdpdef.h>
#include <Ws2bth.h>
//#include <devguid.h>
//#include <setupapi.h>
//#include <cfgmgr32.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <string>

#define _VACT_SIZE_T SIZE_T
#define _VACT_DWORD DWORD
#define _VACT_CHAR char
#define _VACT_BUFFER_SIZE 1

#define _VACT_CMD_SIZE 20
#define _VACT_VALUE_SIZE 16
#define _VACT_DESC_SIZE 10

#include "CoreMinimal.h"
#include "VActDevicesTypes.generated.h"

UENUM()
enum class EDeviceType
{
	None = 0,
	COM = 1 << 0,
	HID = 1 << 1,
	USB = 1 << 2,
	Audio = 1 << 3,
	Bluetooth = 1 << 4,
};

UENUM()
enum class EDilimFlag
{
	None = 0,
	Open = 1 << 0,
	Close = 1 << 1,
	Escape = 1 << 2,
	Count = 1 << 3,
	Context = Open | Close,
	ContextEscape = Open | Close | Escape
};

//USTRUCT()
//struct VACTDEVICES_API FCmd
//{
//	GENERATED_BODY()
//
//	uint16 Flag;
//	uint8 Context;
//	uint8 Code;
//	union
//	{
//		bool B;
//		float F;
//		double D;
//		char C;
//		uint8 UC;
//		int32 I;
//		uint32 UI;
//		int64 L;
//		uint64 UL;
//		bool B4[4];
//		float F4[4];
//		double D4[4];
//		char C4[4];
//		uint8 UC4[4];
//		int32 I4[4];
//		uint32 UI4[4];
//		int64 L2[2];
//		uint64 UL2[2];
//		struct {
//			uint16 Type;
//			uint32 Id;
//			uint32 Size;
//		} Desc;
//		uint8_t _Raw[_VACT_CMD_SIZE];
//	} Value;
//};


USTRUCT()
struct VACTDEVICES_API FDilim
{
	GENERATED_BODY()

	_VACT_CHAR Open;
	_VACT_CHAR Close;
	_VACT_CHAR Escape;
	int32 N;
	EDilimFlag Flag;

	static FORCEINLINE FDilim Count(int32 N) { return { ' ', ' ', ' ', N, EDilimFlag::Count }; }
	static FORCEINLINE FDilim Tuple() { return { '(', ')', ' ', -1, EDilimFlag::Context }; }
	static FORCEINLINE FDilim Object() { return { '{', '}', ' ', -1, EDilimFlag::Context }; }
	static FORCEINLINE FDilim List() { return { '[', ']', ' ', -1, EDilimFlag::Context }; }
	static FORCEINLINE FDilim Tag() { return { '<', '>', ' ', -1, EDilimFlag::Context }; }
	static FORCEINLINE FDilim Line() { return { ' ', '\n', ' ', -1, EDilimFlag::Close }; }
	static FORCEINLINE FDilim Tab() { return { ' ', '\t', ' ', -1, EDilimFlag::Close }; }
	static FORCEINLINE FDilim Space() { return { ' ', ' ', ' ', -1, EDilimFlag::Close }; }
	static FORCEINLINE FDilim Comma() { return { ' ', ',', ' ', -1, EDilimFlag::Close }; }
	static FORCEINLINE FDilim Null() { return { ' ', '\0', ' ', -1, EDilimFlag::Close }; }
	static FORCEINLINE FDilim String() { return { '"', '"', '\\', -1, EDilimFlag::ContextEscape }; }
};

USTRUCT(BlueprintType)
struct VACTDEVICES_API FDevice
{
	GENERATED_BODY()

#if PLATFORM_WINDOWS
	//COMMTIMEOUTS TimeOut;
	HANDLE Handle;
	COMSTAT Status;
	DWORD Error;
	DCB Desc;
#elif PLATFORM_MAC

#elif PLATFORM_LINUX

#endif
	FString SyntaxName;
	FDilim Dilim;
	bool bConnected;
};

USTRUCT(BlueprintType)
struct VACTDEVICES_API FConnect
{
	GENERATED_BODY()

#if PLATFORM_WINDOWS
	SOCKET Socket;
#elif PLATFORM_MAC

#elif PLATFORM_LINUX

#endif
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Name;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Adress;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	bool bConnected;
};


USTRUCT(BlueprintType)
struct VACTDEVICES_API FDeviceInfo
{
	GENERATED_BODY()

#if PLATFORM_WINDOWS
	BLUETOOTH_DEVICE_INFO Info;
#elif PLATFORM_MAC

#elif PLATFORM_LINUX

#endif
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	EDeviceType Type;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString DeviceName;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Adress;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	bool bConnected;
};