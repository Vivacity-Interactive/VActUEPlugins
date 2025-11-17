#include "APISequenceCallback.h"
#include "VActAPI.h"

UAPISequenceCallback::UAPISequenceCallback()
{

}

bool UAPISequenceCallback::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bValid = true;
	for (int32 Index = 0; Index < Sequence.Num(); ++Index)
	{
		bValid = Sequence[Index] != nullptr 
			&& Sequence[Index]->OnDataIn(Request, SelfEntry, Parent, Instance, UserId);

		if (!bValid) { bValid = OnPartial(Index); break; }
	}
	return bValid;
}

bool UAPISequenceCallback::OnDataOut(
	TUniquePtr<FHttpServerResponse>& Response,
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bValid = true;
	for (int32 Index = 0; Index < Sequence.Num(); ++Index)
	{
		bValid = Sequence[Index] != nullptr 
			&& Sequence[Index]->OnDataOut(Response, Request, SelfEntry, Parent, Instance, UserId);

		if (!bValid) { bValid = OnPartial(Index); break; }
	}
	return bValid;
}

bool UAPISequenceCallback::OnPartial_Implementation(
	const int32& ErrorIndex
)
{
	return false;
}