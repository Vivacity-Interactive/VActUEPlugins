#include "APICallback.h"
#include "APIInstance.h"

bool UAPICallback::OnDataIn(
		const FHttpServerRequest& Request,
		const FAPIEntry& SelfEntry,
		UAPIRoute* Parent,
		UAPIInstance* Instance,
		FGuid& UserId
	)
{
	return false;
}

bool UAPICallback::OnDataOut(
		TUniquePtr<FHttpServerResponse>& Response,
		const FHttpServerRequest& Request,
		const FAPIEntry& SelfEntry,
		UAPIRoute* Parent,
		UAPIInstance* Instance,
		FGuid& UserId
	)
{
	return false;
}