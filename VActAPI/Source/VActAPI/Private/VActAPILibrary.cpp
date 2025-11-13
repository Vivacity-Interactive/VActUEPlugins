// Copyright Epic Games, Inc. All Rights Reserved.

#include "VActAPILibrary.h"
#include "VActAPI.h"

UVActAPILibrary::UVActAPILibrary(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{

}

float UVActAPILibrary::VActAPISampleFunction(float Param)
{
	return -1;
}


// API Functions

bool UVActAPILibrary::EntryAdd(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, UPARAM(ref) FAPIEntry& InEntry, FName& OutName)
{
	return FVActAPI::Entry(InAPIInstance, InRoute, InEntry, OutName);
}
