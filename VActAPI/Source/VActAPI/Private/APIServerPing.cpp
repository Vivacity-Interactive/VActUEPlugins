#include "APIServerPing.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

UAPIServerPing::UAPIServerPing()
	: bPingScopeId(true)
	, bAsHeader(false)
	, Value("a1f5c381-a0dc-48f0-97dc-3d34e4c59b79")
	, Key("X-Scope-Id")
{

}

bool UAPIServerPing::OnDataIn(
		const FHttpServerRequest& Request,
		const FAPIEntry& SelfEntry,
		UAPIRoute* Parent,
		UAPIInstance* Instance,
		FGuid& UserId
	)
{
	bool bSuccess = true;
	FString _Value = TEXT("");
	uint32 Ip = -1;
	int32 Port = -1;
	if (bAsHeader)
	{
		const TArray<FString>* Values = Request.Headers.Find(Key);
		bSuccess = Values != nullptr && Values->Num() > 0;
		if (bSuccess) { (_Value = (*Values)[0]); }
	}
	else
	{
		FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(Request.Body.GetData()), Request.Body.Num());
		_Value = FString(Converter.Length(), Converter.Get());
		
	}

	if (Request.PeerAddress.IsValid())
	{
		Request.PeerAddress->GetIp(Ip);
		Port = Request.PeerAddress->GetPort();
	}
	
	bSuccess &= OnPingIn(_Value, static_cast<int64>(Ip), Port);
	return bSuccess;
}

bool UAPIServerPing::OnDataOut(
		TUniquePtr<FHttpServerResponse>& Response,
		const FHttpServerRequest& Request,
		const FAPIEntry& SelfEntry,
		UAPIRoute* Parent,
		UAPIInstance* Instance,
		FGuid& UserId
	)
{
	bool bRespond = Response.IsValid();
	int64 Ip = -1;
	int32 Port = -1;
	if (bRespond)
	{
		if (bAsHeader)
		{
			Response->Headers.Add(Key, { bPingScopeId ? Instance->Identity.ScopeUnique : *Value });
		}
		else
		{
			FTCHARToUTF8 Converter(bPingScopeId ? Instance->Identity.ScopeUnique : *Value);
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

		bRespond &= OnPingOut(Ip, Port);
	}
	return bRespond;
}

bool UAPIServerPing::OnPingIn_Implementation(
		const FString& InValue,
		int64 Ip,
		int32 Port
	)
{
	return true;
}

bool UAPIServerPing::OnPingOut_Implementation(
		int64 Ip,
		int32 Port
	)
{
	return true;
}