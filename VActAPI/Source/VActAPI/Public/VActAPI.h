#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "Misc/Guid.h"

#include "HttpServerModule.h"
#include "IHttpRouter.h"
#include "HttpServerResponse.h"
#include "HttpServerRequest.h"
#include "HttpPath.h"

#include "ImageCore.h"
#include "PixelFormat.h"
#include "IImageWrapper.h"

#include "CoreMinimal.h"
#include "VActAPI.generated.h"

class UAPIInstance;

USTRUCT()
struct VACTAPI_API FVActAPI
{
	GENERATED_BODY()

	static const FString AuthorizationKey;

	static const FString SessionPathTokenKey;

	static const FString RequestCodeKey;

	const static TMap<EAPIImageFormat, EImageFormat> ImageFormatMapE;

	const static TMap<EAPIGammaSpace, EGammaSpace> GammaSpaceMapE;

	const static TMap<EAPIImageRawFormat, EPixelFormat> ImageRawFormatMapE;

	const static TMap<ERawImageFormat::Type, EPixelFormat> _ImageRawFormatMapE;

	const static TMap<EPixelFormat, ERawImageFormat::Type> _ImageRawFormatMapInvE;

	const static TMap<EAPIEntryMode, FString> EntryModeMap;

	const static TMap<FString, EAPIEntryMode> EntryModeMapInv;

	const static TMap<EAPIEntryMode, EHttpServerRequestVerbs> EntryModeMapE;

	const static TMap<EAPIEntryContent, FString> EntryContentMap;

	const static TMap<FString, EAPIEntryContent> EntryContentMapInv;

	const static TMap<EAPIImageFormat, EAPIEntryContent> EntryImageContentMapE;

	FORCEINLINE static bool Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName, FHttpRouteHandle& Handle)
	{
		const bool bValid = InAPIInstance != nullptr && InRoute != nullptr;
		return bValid && _Unsafe_Entry(InAPIInstance, InRoute, InEntry, OutName, Handle);
	}

	FORCEINLINE static bool Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName)
	{
		FHttpRouteHandle Handle;
		const bool bValid = InAPIInstance != nullptr && InRoute != nullptr;
		return bValid && _Unsafe_Entry(InAPIInstance, InRoute, InEntry, OutName, Handle);
	}

	FORCEINLINE static bool Certificate(UAPIInstance* InAPIInstance)
	{
		const bool bValid = InAPIInstance != nullptr;
		return bValid && _Unsafe_Certificate(InAPIInstance);
	}

	FORCEINLINE static bool Multipart(const FHttpServerRequest& Request, TMap<FString, FAPIConstFormEntry>& Map)
	{
		const TArray<FString>* ContentType = Request.Headers.Find("Content-Type");
		const bool bValid = ContentType != nullptr;
		return bValid && _Unsafe_Multipart(Request, Map);
	}

	static bool _Unsafe_Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName, FHttpRouteHandle& Handle);

	static bool _Unsafe_Certificate(UAPIInstance* InAPIInstance);

	static bool _Unsafe_Multipart(const FHttpServerRequest& Request, TMap<FString, FAPIConstFormEntry>& Map);

	static bool _Unsafe_Multipart(const TArrayView<const uint8>& Segment, FAPIConstFormEntry& Entry);

	static bool _Unsafe_Multipart(const FHttpServerRequest& Request, FAPIConstFormEntry& Entry);

	static bool _Unsafe_Multipart(const FHttpServerRequest& Request, const FString& Target, TArray<FAPIConstFormEntry>& Entries);

	static bool _Unsafe_Token(const uint8* Buffer, int32 NumBuffer, const uint8* Token, const int32 NumToken, int32& Pivot, bool bConsume = true);

	static bool _Unsafe_TokenNot(const uint8* Buffer, int32 NumBuffer, const uint8* Token, const int32 NumToken, int32& Pivot, bool bConsume = true);

	static bool _Unsafe_TokenRangeNot(const uint8* Buffer, int32 NumBuffer, uint8 From, uint8 To, int32& Pivot, bool bConsume = true);

};
