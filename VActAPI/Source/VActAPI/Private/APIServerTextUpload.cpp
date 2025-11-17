#include "APIServerTextUpload.h"

bool UAPIServerTextUpload::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	TArray<FAPIConstMultipartSegment> FormSegments;
	bool bSuccess = Request.Body.Num() > 0 && FVActAPI::Multipart(Request, FormSegments) && FormSegments.Num() > 0;
	
#if WITH_EDITOR
	UE_LOG(LogTemp, Warning, TEXT("%s try starting text request"), *GetNameSafe(this));
#endif

	if (bSuccess)
	{
		FAPIConstMultipartSegment& Segment = FormSegments[0];

		FUTF8ToTCHAR Converter((const ANSICHAR*)Segment.Body.GetData(), Segment.Body.Num());
		FString Text(Converter.Length(), Converter.Get());
		bSuccess &= OnTextIn(Text);
	}
	return bSuccess;
}

bool UAPIServerTextUpload::OnDataOut(
	TUniquePtr<FHttpServerResponse>& Response,
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bSuccess = Response.IsValid();
	if (bSuccess)
	{
		FString Text = OnTextOut(Text);
		FTCHARToUTF8 Converter(*Text);
		Response->Body.SetNumUninitialized(Converter.Length());
		FMemory::Memcpy(Response->Body.GetData(), Converter.Get(), Converter.Length());
	}
	return bSuccess;
}

bool UAPIServerTextUpload::OnTextIn_Implementation(
	const FString& Text
)
{

	return true;
}

FString UAPIServerTextUpload::OnTextOut_Implementation(
	const FString& AssetId
)
{
	return TEXT("");
}