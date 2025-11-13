#include "APIServerLogin.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

UAPIServerLogin::UAPIServerLogin()
	: bFixedToken(false)
	, bAsHeader(true)
	, Token("d72ddb72-ac93-42be-8767-db6ea5242373")
	, Key("X-Bearer-Token")
{

}

bool UAPIServerLogin::OnDataIn(
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

	bSuccess &= OnLoginIn(Value, static_cast<int64>(Ip), Port);
	return bSuccess;
}

bool UAPIServerLogin::OnDataOut(
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
	
	int64 Ip = -1;
	int32 Port = -1;
	if (bRespond)
	{
		_Token = bFixedToken ? Token : UserId.ToString();
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

		bRespond &= OnLoginOut(Ip, Port);
	}
	return bRespond;
}

bool UAPIServerLogin::OnLoginIn_Implementation(
	const FString& InToken,
	int64 Address,
	int32 Port
)
{
	return true;
}

bool UAPIServerLogin::OnLoginOut_Implementation(
	int64 Address,
	int32 Port
)
{
	return true;
}