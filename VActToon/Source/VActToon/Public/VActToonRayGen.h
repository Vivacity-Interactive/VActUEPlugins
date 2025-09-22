#pragma once

#include "CoreMinimal.h"

#include "RenderGraphUtils.h"
#include "Engine/TextureRenderTargetVolume.h"
#include "Runtime/Engine/Classes/Engine/TextureRenderTarget2D.h"

class FRayTracingScene;

struct  FVActToonRayGenParameters
{

	FIntPoint GetRenderTargetSize() const
	{
		return CachedRenderTargetSize;
	}

	FVActToonRayGenParameters() {}; // consider delete this, otherwise the target size will not be set, or just add setter
	FVActToonRayGenParameters(UTextureRenderTarget2D* IORenderTarget)
		: RenderTarget(IORenderTarget)
	{
		CachedRenderTargetSize = RenderTarget ? FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY) : FIntPoint::ZeroValue;
	}

	UTextureRenderTarget2D* RenderTarget;
	FRayTracingScene* Scene;
	FIntPoint CachedRenderTargetSize;
};

class VACTTOON_API FVActToonRayGen
{
public:
	FVActToonRayGen();

	void BeginRendering();
	void EndRendering();
	void UpdateParameters(FVActToonRayGenParameters& DrawParameters);
private:
	void Execute_RenderThread(FPostOpaqueRenderParameters& Parameters);

	/// The delegate handle to our function that will be executed each frame by the renderer
	FDelegateHandle PostOpaqueRenderDelegate;
	/// Cached Shader Manager Parameters
	FVActToonRayGenParameters CachedParams;
	/// Whether we have cached parameters to pass to the shader or not
	volatile bool bCachedParamsAreValid;

	/// We create the shader's output texture and UAV and save to avoid reallocation
	FTextureRHIRef ShaderOutputTexture;
	FUnorderedAccessViewRHIRef ShaderOutputTextureUAV;
};