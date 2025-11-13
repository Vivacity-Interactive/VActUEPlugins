#include "APIServerLogout.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

UAPIServerLogout::UAPIServerLogout()
	: bFixedToken(false)
	, bAsHeader(true)
	, Token("74850b0e-1c8d-4163-9060-3530f64fc834")
	, Key("X-Bearer-Token")
{

}

bool UAPIServerLogout::OnDataIn(
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

	bSuccess &= OnLogoutIn(Value, static_cast<int64>(Ip), Port);
	return bSuccess;
}

bool UAPIServerLogout::OnDataOut(
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
		_Token = bFixedToken ? Token : UserId.ToString();
		Instance->CommitUser(UserId, true);
		bRespond &= OnLogoutOut(Ip, Port);
	}
	return bRespond;
}

bool UAPIServerLogout::OnLogoutIn_Implementation(
	const FString& InToken,
	int64 Address,
	int32 Port
)
{
	return true;
}

bool UAPIServerLogout::OnLogoutOut_Implementation(
	int64 Address,
	int32 Port
)
{
	return true;
}