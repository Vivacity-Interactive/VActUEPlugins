#include "APIServerImageUploadPrompt.h"
#include "APIInstance.h"
#include "Misc/Guid.h"
#include "VActAPI.h"

const FString UAPIServerImageUploadPrompt::DefaultPrompt = R"(
<!DOCTYPE html>
<html>
  <body>
    <h1>Upload a File</h1>
    <form action="{{ActionUrl}}" method="post" enctype="{{ContentType}}" accept="{{AcceptFormats}}">
        <input type="file" name="{{Key}}" />
        <input type="submit" value="Upload File" />
    </form>
  </body>
</html>
)";

UAPIServerImageUploadPrompt::UAPIServerImageUploadPrompt()
	: bSecurityIntersect(true)
	, bUseFile(false)
	, File()
	, Prompt(DefaultPrompt)
	, Key(TEXT("File"))
	, ActionEntry()
{

}

bool UAPIServerImageUploadPrompt::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bSuccess = false;

	FString _Prompt = TEXT("");
	if (Request.Body.Num())
	{
		FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(Request.Body.GetData()), Request.Body.Num());
		_Prompt = FString(Converter.Length(), Converter.Get());
	}

	bSuccess &= OnPromptIn(_Prompt);
	return bSuccess;
}

bool UAPIServerImageUploadPrompt::OnDataOut(
	TUniquePtr<FHttpServerResponse>& Response,
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	bool bRespond = Response.IsValid();
	if (bRespond)
	{
		FString AcceptFormats;
		TArray<FString> _SupportedFormats;
		int32 FormatCount = SupportedFormats.Num();
		if (FormatCount > 0)
		{
			for (const auto& Format : SupportedFormats)
			{
				_SupportedFormats.Add(FVActAPI::EntryContentMap[Format]);
			}
			AcceptFormats = FString::Join(_SupportedFormats, TEXT(","));
		}
		else
		{
			AcceptFormats = TEXT("*/*");
		}
		
		_SupportedFormats.SetNumUninitialized(FormatCount);

		if (bSecurityIntersect) { SelfEntry.SecurityIntersectInto(ActionEntry); }

		FString SessionPathToken = Request.PathParams.Contains(FVActAPI::SessionPathTokenKey)
			? Request.PathParams[FVActAPI::SessionPathTokenKey]
			: TEXT("");

		FString RequestCodeToken = Request.PathParams.Contains(FVActAPI::RequestCodeKey)
			? Request.PathParams[FVActAPI::RequestCodeKey]
			: TEXT("");
		
		FString _Prompt = Prompt
			.Replace(TEXT("{{ContentType}}"), *FVActAPI::EntryContentMap[ActionEntry.Content])
			.Replace(TEXT("{{ActionUrl}}"), *ActionEntry.GetEntryUrl(Instance->Identity, SessionPathToken, RequestCodeToken, true, true))
			.Replace(TEXT("{{AcceptFormats}}"), *AcceptFormats)
			.Replace(TEXT("{{Key}}"), *Key);

		bRespond &= OnPromptOut(_Prompt);

		if (bRespond)
		{
			FTCHARToUTF8 Converter(*_Prompt);
			Response->Body.SetNumUninitialized(Converter.Length());
			FMemory::Memcpy(Response->Body.GetData(), Converter.Get(), Converter.Length());
		}
	}
	return bRespond;
}

bool UAPIServerImageUploadPrompt::OnPromptIn_Implementation(
	const FString& InPrompt
)
{
	return true;
}

bool UAPIServerImageUploadPrompt::OnPromptOut_Implementation(
	FString& InPrompt
)
{
	return true;
}

#if WITH_EDITOR
void UAPIServerImageUploadPrompt::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	FName PropertyName = PropertyChangedEvent.Property ? PropertyChangedEvent.Property->GetFName() : NAME_None;
	if (PropertyName == GET_MEMBER_NAME_CHECKED(UAPIServerImageUploadPrompt, SelectedSupportedFormats))
	{
		SupportedFormats.Empty();
		SupportedFormats.Append(SelectedSupportedFormats);
	}
}
#endif