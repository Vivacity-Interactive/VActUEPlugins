#include "APISwitchCallback.h"
#include "VActAPI.h"

UAPISwitchCallback::UAPISwitchCallback()
{

}

bool UAPISwitchCallback::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	int32 Index = -1;
	EAPIEntryContent Content = EAPIEntryContent::Unknown;

	const TArray<FString>* ContentType = Request.Headers.Find("Content-Type");
	const bool bValid = ContentType != nullptr && ContentType->Num() > 0;
	if (bValid) { Content = FVActAPI::EntryContentMapInv[(*ContentType)[0]]; }

	return bValid
		&& OnSwitch(Content,Index)
		&& Cases.IsValidIndex(Index) 
		&& Cases[Index] != nullptr 
		&& Cases[Index]->OnDataIn(Request, SelfEntry, Parent, Instance, UserId);
}

bool UAPISwitchCallback::OnDataOut(
	TUniquePtr<FHttpServerResponse>& Response,
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	int32 Index = -1;
	EAPIEntryContent Content = EAPIEntryContent::Unknown;

	const TArray<FString>* ContentType = Request.Headers.Find("Content-Type");
	const bool bValid = ContentType != nullptr && ContentType->Num() > 0;
	if (bValid) { Content = FVActAPI::EntryContentMapInv[(*ContentType)[0]]; }

	return bValid
		&& OnSwitch(Content, Index)
		&& Cases.IsValidIndex(Index)
		&& Cases[Index] != nullptr
		&& Cases[Index]->OnDataOut(Response, Request, SelfEntry, Parent, Instance, UserId);
}

bool UAPISwitchCallback::OnSwitch_Implementation(
	EAPIEntryContent Content,
	int32& CaseIndex
)
{
	CaseIndex = 0;
	return true;
}