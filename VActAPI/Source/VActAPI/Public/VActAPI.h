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

	static const TArray<uint8> MPFD_Ctx;

	static const TArray<uint8> MPFD_Hdr;

	static const TArray<uint8> MPFD_Ign;

	static const uint8 MPFD_Prt;

	static const uint8 MPFD_Val;

	static const uint8 MPFD_Key;

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

	FORCEINLINE static bool Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName, FHttpRouteHandle& Handle, UObject* ContextObject = nullptr)
	{
		const bool bValid = InAPIInstance != nullptr && InRoute != nullptr;
		return bValid && _Unsafe_Entry(InAPIInstance, InRoute, InEntry, OutName, Handle, ContextObject);
	}

	FORCEINLINE static bool Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName, UObject* ContextObject = nullptr)
	{
		FHttpRouteHandle Handle;
		const bool bValid = InAPIInstance != nullptr && InRoute != nullptr;
		return bValid && _Unsafe_Entry(InAPIInstance, InRoute, InEntry, OutName, Handle, ContextObject);
	}

	FORCEINLINE static bool Certificate(UAPIInstance* InAPIInstance)
	{
		const bool bValid = InAPIInstance != nullptr;
		return bValid && _Unsafe_Certificate(InAPIInstance);
	}

	FORCEINLINE static bool Multipart(const FHttpServerRequest& Request, TArray<FAPIConstMultipartSegment>& Segments)
	{
		const TArray<FString>* ContentType = Request.Headers.Find("Content-Type");
		const bool bValid = ContentType != nullptr;
		return bValid && _Unsafe_Multipart(Request, Segments);
	}

	static bool _Unsafe_Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName, FHttpRouteHandle& Handle, UObject* ContextObject = nullptr);

	static bool _Unsafe_Certificate(UAPIInstance* InAPIInstance);

	static bool _Unsafe_KeyValue(TArrayView<const uint8>& Property, TArrayView<const uint8>& Key, TArrayView<const uint8>& Value, bool& bKeyOnly);
	
	static bool _Unsafe_Properties(TArrayView<const uint8>& Header, TArray<TArrayView<const uint8>>& OutProperties);

	static bool _Unsafe_Headers(TArrayView<const uint8>& Headers, TMap<FString, TArray<TArrayView<const uint8>>>& OutHeaders);

	static bool _Unsafe_Multipart(const FHttpServerRequest& Request, TArray<FAPIConstMultipartSegment>& Segments);

	static bool _Unsafe_Token(const uint8* Buffer, int32 NumBuffer, const uint8* Token, const int32 NumToken, int32& Pivot, bool bConsume = true);

	static bool _Unsafe_TokenFirst(const uint8* Buffer, int32 NumBuffer, const uint8* Token, const int32 NumToken, int32& Pivot, bool bConsume = true, bool bResetPivot = true);

	static bool _Unsafe_TokenFirst(const uint8* Buffer, int32 NumBuffer, const uint8 Token, int32& Pivot, bool bConsume = true);

	static bool _Unsafe_TokenSkip(const uint8* Buffer, int32 NumBuffer, const uint8 Range[2], int32& Pivot);

	static bool _Unsafe_TokenSkipNot(const uint8* Buffer, int32 NumBuffer, const uint8 Range[2], int32& Pivot);

};
