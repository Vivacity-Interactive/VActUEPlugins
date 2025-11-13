#include "APIServerTokenRenew.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

UAPIServerTokenRenew::UAPIServerTokenRenew()
	: bFixedToken(false)
	, bAsHeader(true)
	, Token("74850b0e-1c8d-4163-9060-3530f64fc834")
	, Key("X-Bearer-Token")
{

}

bool UAPIServerTokenRenew::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bSuccess = true;
	FString _Token = TEXT("");
	uint32 Ip = -1;
	int32 Port = -1;
	if (bAsHeader)
	{
		const TArray<FString>* Values = Request.Headers.Find(Key);
		bSuccess = Values != nullptr && Values->Num() > 0;
		if (bSuccess) { (_Token = (*Values)[0]); }
	}
	else
	{
		FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(Request.Body.GetData()), Request.Body.Num());
		FString Value = FString(Converter.Length(), Converter.Get());
	}

	if (Request.PeerAddress.IsValid())
	{
		Request.PeerAddress->GetIp(Ip);
		Port = Request.PeerAddress->GetPort();
	}

	bSuccess &= OnRenewIn(_Token, static_cast<int64>(Ip), Port);
	return bSuccess;
}

bool UAPIServerTokenRenew::OnDataOut(
	TUniquePtr<FHttpServerResponse>& Response,
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bRespond = Response.IsValid();
	FString _Token = TEXT("");

	bRespond &= Instance->Keeper(_Token, Token, bFixedToken, UserId);

	if (!bFixedToken) {  }
	else { FString _Token = Token; }

	int64 Ip = -1;
	int32 Port = -1;
	if (bRespond)
	{
		if (bAsHeader)
		{
			Response->Headers.Add(Key, { _Token });
		}
		else
		{
			FTCHARToUTF8 Converter(_Token);
			Response->Body.SetNumUninitialized(Converter.Length());
			FMemory::Memcpy(Response->Body.GetData(), Converter.Get(), Converter.Length());
		}

		if (Request.PeerAddress.IsValid())
		{
			uint32 _Ip;
			Request.PeerAddress->GetIp(_Ip);
			Port = Request.PeerAddress->GetPort();
			Ip = static_cast<int64>(_Ip);
		}

		bRespond &= OnRenewOut(Ip, Port);
	}
	return bRespond;
}

bool UAPIServerTokenRenew::OnRenewIn_Implementation(
	const FString& InToken,
	int64 Ip,
	int32 Port
)
{
	return true;
}

bool UAPIServerTokenRenew::OnRenewOut_Implementation(
	int64 Ip,
	int32 Port
)
{
	return true;
}