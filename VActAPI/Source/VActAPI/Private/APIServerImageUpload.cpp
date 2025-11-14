#include "APIServerImageUpload.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

#include "RHICommandList.h"
#include "RenderUtils.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"

UAPIServerImageUpload::UAPIServerImageUpload()
	: bForceImageRawFormat(false)
	, ImageFormat(EAPIImageFormat::Png)
	, RawImageFormat(EAPIImageRawFormat::BGRA8)
	, GammaSpace(EAPIGammaSpace::sRGB)
{

}

bool UAPIServerImageUpload::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	TMap<FString, FAPIConstFormEntry> Form;
	bool bSuccess = Request.Body.Num() > 0 && FVActAPI::Multipart(Request, Form);
	UTexture2D* Image = nullptr;

#if WITH_EDITOR
	UE_LOG(LogTemp, Warning, TEXT("%s try starting image request"), *GetNameSafe(this));
#endif

	if (bSuccess)
	{
		FImage _Image;
		IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
		EImageFormat Format = ImageWrapperModule.DetectImageFormat(Request.Body.GetData(), Request.Body.Num());
#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("%s starting image request"), *GetNameSafe(this));
#endif
		const bool bValid = Format != EImageFormat::Invalid;
		if (!bValid) { return false; }

#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("%s format image valid"), *GetNameSafe(this));
#endif
		const bool bDecompressed = ImageWrapperModule.DecompressImage(Request.Body.GetData(), Request.Body.Num(), _Image);
		if (!bDecompressed) { return false; }

#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("%s image decompressed"), *GetNameSafe(this));
#endif
		EPixelFormat _PixelFormat = bForceImageRawFormat
			? FVActAPI::ImageRawFormatMapE[RawImageFormat]
			: FVActAPI::_ImageRawFormatMapE[_Image.Format];

		Image = UTexture2D::CreateTransient(_Image.SizeX, _Image.SizeY, _PixelFormat);
#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("%s image texture created"), *GetNameSafe(this));
#endif
	}

	bSuccess &= Image != nullptr && OnImageIn(Image);
	return bSuccess;
}

bool UAPIServerImageUpload::OnDataOut(
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
		FString AssetId = TEXT("");
		UTexture2D* Image = nullptr;

		bRespond &= !AssetId.IsEmpty() && (Image = OnImageOut(AssetId)) != nullptr;

		if (bRespond)
		{
			EGammaSpace _GammaSpace = bForceImageRawFormat
				? FVActAPI::GammaSpaceMapE[GammaSpace]
				: (Image->SRGB ? EGammaSpace::sRGB : EGammaSpace::Linear);

			FTextureResource * ResourceData = Image->GetResource();
			FTexturePlatformData* PlatformData = Image->GetPlatformData();
			
			FImage _Image;

#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("%s starting image response"), *GetNameSafe(this));
#endif

			const bool bGPUImage = ResourceData != nullptr;
			const bool bCPUImage = PlatformData != nullptr;
			bRespond &= bGPUImage || bCPUImage;
			if (bCPUImage)
			{
				const bool bValid = PlatformData != nullptr && PlatformData->Mips.Num() <= 0;

				FTexture2DMipMap& Mip = Image->GetPlatformData()->Mips[0];
				void* _Data = Mip.BulkData.Lock(LOCK_READ_ONLY);

				ERawImageFormat::Type _RawImageFormat = bForceImageRawFormat
					? FVActAPI::_ImageRawFormatMapInvE[FVActAPI::ImageRawFormatMapE[RawImageFormat]]
					: FVActAPI::_ImageRawFormatMapInvE[PlatformData->PixelFormat];

				_Image.Init(Mip.SizeX, Mip.SizeY, _RawImageFormat, FVActAPI::GammaSpaceMapE[GammaSpace]);
				FMemory::Memcpy(_Image.RawData.GetData(), _Data, _Image.RawData.Num());
				Mip.BulkData.Unlock();
			}
			else if (bGPUImage)
			{
				TArray<FColor> Pixels;
				FTextureRHIRef ImageHRI = ResourceData->GetTexture3DRHI();
				
				const bool bValid = ImageHRI.IsValid();
				if (!bValid) { return false; }

				const int32 SizeX = Image->GetSizeX();
				const int32 SizeY = Image->GetSizeY();
				Pixels.SetNumUninitialized(SizeX * SizeY);

				ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
					[ImageHRI, SizeX, SizeY, &Pixels](FRHICommandListImmediate& RHICmdList)
					{
						FReadSurfaceDataFlags ReadFlags(RCM_UNorm, CubeFace_MAX);
						RHICmdList.ReadSurfaceData(
							ImageHRI,
							FIntRect(0, 0, SizeX, SizeY),
							Pixels,
							FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX)
						);
					});

				ERawImageFormat::Type _RawImageFormat = bForceImageRawFormat
					? FVActAPI::_ImageRawFormatMapInvE[FVActAPI::ImageRawFormatMapE[RawImageFormat]]
					: FVActAPI::_ImageRawFormatMapInvE[ImageHRI->GetFormat()];

				_Image.Init(SizeX, SizeY, _RawImageFormat, FVActAPI::GammaSpaceMapE[GammaSpace]);
				FMemory::Memcpy(_Image.RawData.GetData(), Pixels.GetData(), _Image.RawData.Num());
			}

			if (bRespond)
			{
				FImageView ImageView(_Image);
				IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
				TArray64<uint8> Compressed;
				ImageWrapperModule.CompressImage(Compressed, FVActAPI::ImageFormatMapE[ImageFormat], ImageView);

				Response->Body.SetNumUninitialized(Compressed.Num());
				FMemory::Memcpy(Response->Body.GetData(), Compressed.GetData(), Compressed.Num());
			}
			
		}
	}
	return bRespond;
}

bool UAPIServerImageUpload::OnImageIn_Implementation(
	UTexture2D* Image
)
{
	return true;
}

UTexture2D* UAPIServerImageUpload::OnImageOut_Implementation(
	const FString& AssetId
)
{
	return nullptr;
}