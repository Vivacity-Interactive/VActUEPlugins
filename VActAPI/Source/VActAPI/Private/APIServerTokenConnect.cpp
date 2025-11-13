#include "APIServerTokenConnect.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

UAPIServerTokenConnect::UAPIServerTokenConnect()
	: bFixedToken(false)
	, bAsHeader(true)
	, Token("74850b0e-1c8d-4163-9060-3530f64fc834")
	, Key("X-Bearer-Token")
{

}

bool UAPIServerTokenConnect::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bSuccess = true;
	FString Value = TEXT("");
	uint32 Ip = -1;
	int32 Port = -1;
	if (bAsHeader)
	{
		const TArray<FString>* Values = Request.Headers.Find(Key);
		bSuccess = Values != nullptr && Values->Num() > 0;
		if (bSuccess) { (Value = (*Values)[0]); }
	}
	else
	{
		FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(Request.Body.GetData()), Request.Body.Num());
		Value = FString(Converter.Length(), Converter.Get());
	}

	if (Request.PeerAddress.IsValid())
	{
		Request.PeerAddress->GetIp(Ip);
		Port = Request.PeerAddress->GetPort();
	}

	bSuccess &= OnConnectIn(Value, static_cast<int64>(Ip), Port);
	return bSuccess;
}

bool UAPIServerTokenConnect::OnDataOut(
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
	
	bRespond &= Instance->Session(_Token, Token, bFixedToken, UserId);
	
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

		bRespond &= OnConnectOut(Ip, Port);
	}
	return bRespond;
}

bool UAPIServerTokenConnect::OnConnectIn_Implementation(
	const FString& InToken,
	int64 Address,
	int32 Port
)
{
	return true;
}

bool UAPIServerTokenConnect::OnConnectOut_Implementation(
	int64 Address,
	int32 Port
)
{
	return true;
}