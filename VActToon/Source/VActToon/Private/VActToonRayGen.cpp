#include "VActToonRayGen.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderTargetPool.h"
#include "RHI.h"
#include "Modules/ModuleManager.h"
#include "RayTracingDefinitions.h"
#include "RayTracingPayloadType.h"
#include "../Private/RayTracing/RayTracingScene.h"
#include "../Private/SceneRendering.h"
#include "RenderGraphUtils.h"

#define NUM_THREADS_PER_GROUP_DIMENSION 8

class FVActToonRayGenRGS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVActToonRayGenRGS)
	SHADER_USE_ROOT_PARAMETER_STRUCT(FVActToonRayGenRGS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWTexture2D<float4>, outTex)
		SHADER_PARAMETER_RDG_BUFFER_SRV(RaytracingAccelerationStructure, TLAS)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 /*PermutationId*/)
	{
		return ERayTracingPayloadType::Minimal;
	}

	static inline void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

		//We're using it here to add some preprocessor defines. That way we don't have to change both C++ and HLSL code when we change the value for NUM_THREADS_PER_GROUP_DIMENSION
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_X"), NUM_THREADS_PER_GROUP_DIMENSION);
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_Y"), NUM_THREADS_PER_GROUP_DIMENSION);
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_Z"), NUM_THREADS_PER_GROUP_DIMENSION);
	}
};
IMPLEMENT_GLOBAL_SHADER(FVActToonRayGenRGS, "/Plugin/Shaders/Private/VActToonRayTrace.usf", "VActToonRayGenRGS", SF_RayGen);

class FVActToonRayGenCHS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVActToonRayGenCHS)
	SHADER_USE_ROOT_PARAMETER_STRUCT(FVActToonRayGenCHS, FGlobalShader)

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 PermutationId)
	{
		return FVActToonRayGenRGS::GetRayTracingPayloadType(PermutationId);
	}

	using FParameters = FEmptyShaderParameters;
};
IMPLEMENT_GLOBAL_SHADER(FVActToonRayGenCHS, "/Plugin/CustomShaders/VActToonRayTrace.usf", "closestHit=VActToonRayGenCHS", SF_RayHitGroup);

class FVActToonRayGenMS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FVActToonRayGenMS)
	SHADER_USE_ROOT_PARAMETER_STRUCT(FVActToonRayGenMS, FGlobalShader)

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 PermutationId)
	{
		return FVActToonRayGenRGS::GetRayTracingPayloadType(PermutationId);
	}

	using FParameters = FEmptyShaderParameters;
};
IMPLEMENT_GLOBAL_SHADER(FVActToonRayGenMS, "/Plugin/CustomShaders/VActToonRayTrace.usf", "VActToonRayGenMS", SF_RayMiss);

FVActToonRayGen::FVActToonRayGen()
{
}

void FVActToonRayGen::BeginRendering()
{
	//If the handle is already initialized and valid, no need to do anything
	if (PostOpaqueRenderDelegate.IsValid())
	{
		return;
	}
	//Get the Renderer Module and add our entry to the callbacks so it can be executed each frame after the scene rendering is done
	const FName RendererModuleName("Renderer");
	IRendererModule* RendererModule = FModuleManager::GetModulePtr<IRendererModule>(RendererModuleName);
	if (RendererModule)
	{
		PostOpaqueRenderDelegate = RendererModule->RegisterPostOpaqueRenderDelegate(FPostOpaqueRenderDelegate::CreateRaw(this, &FVActToonRayGen::Execute_RenderThread));
	}

	// create output texture
	FIntPoint TextureSize = { CachedParams.RenderTarget->SizeX, CachedParams.RenderTarget->SizeY };
	FRHITextureCreateDesc TextureDesc = FRHITextureCreateDesc::Create2D(TEXT("RaytracingToonOutput"), TextureSize.X, TextureSize.Y, CachedParams.RenderTarget->GetFormat());
	TextureDesc.AddFlags(TexCreate_ShaderResource | TexCreate_UAV);
	ShaderOutputTexture = RHICreateTexture(TextureDesc);
	ENQUEUE_RENDER_COMMAND(CreateUAVCmd)(
		[this](FRHICommandListImmediate& RHICmdList)
		{
			ShaderOutputTextureUAV = RHICmdList.CreateUnorderedAccessView(ShaderOutputTexture, 0);
		});
}

//Stop the compute shader execution
void FVActToonRayGen::EndRendering()
{
	//If the handle is not valid then there's no cleanup to do
	if (!PostOpaqueRenderDelegate.IsValid())
	{
		return;
	}
	//Get the Renderer Module and remove our entry from the PostOpaqueRender callback
	const FName RendererModuleName("Renderer");
	IRendererModule* RendererModule = FModuleManager::GetModulePtr<IRendererModule>(RendererModuleName);
	if (RendererModule)
	{
		RendererModule->RemovePostOpaqueRenderDelegate(PostOpaqueRenderDelegate);
	}

	PostOpaqueRenderDelegate.Reset();
}

void FVActToonRayGen::UpdateParameters(FVActToonRayGenParameters& DrawParameters)
{
	CachedParams = DrawParameters;
	bCachedParamsAreValid = true;
}

void FVActToonRayGen::Execute_RenderThread(FPostOpaqueRenderParameters& Parameters)
#if RHI_RAYTRACING
{
	FRDGBuilder* GraphBuilder = Parameters.GraphBuilder;
	FRHICommandListImmediate& RHICmdList = GraphBuilder->RHICmdList;
	//If there's no cached parameters to use, skip
	//If no Render Target is supplied in the cachedParams, skip
	if (!(bCachedParamsAreValid && CachedParams.RenderTarget))
	{
		return;
	}

	//Render Thread Assertion
	check(IsInRenderingThread());

	// set shader parameters
	FVActToonRayGenRGS::FParameters* PassParameters = GraphBuilder->AllocParameters<FVActToonRayGenRGS::FParameters>();
	PassParameters->ViewUniformBuffer = Parameters.View->ViewUniformBuffer;
	PassParameters->TLAS = CachedParams.Scene->GetLayerView(ERayTracingSceneLayer::Base);
	PassParameters->outTex = ShaderOutputTextureUAV;

	// define render pass needed parameters
	TShaderMapRef<FVActToonRayGenRGS> VActToonRayGenRGS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	FIntPoint TextureSize = { CachedParams.RenderTarget->SizeX, CachedParams.RenderTarget->SizeY };
	FRHIRayTracingScene* RHIScene = CachedParams.Scene->GetRHIRayTracingScene(ERayTracingSceneLayer::Base);

	// add the ray trace dispatch pass
	GraphBuilder->AddPass(
		RDG_EVENT_NAME("VActToonRayGen"),
		PassParameters,
		ERDGPassFlags::Compute,
		[PassParameters, VActToonRayGenRGS, TextureSize, RHIScene, Scene = CachedParams.Scene](FRHIRayTracingCommandList& RHICmdList)
		{
			FRayTracingShaderBindingTableInitializer Initializer;
			Initializer.NumMissShaderSlots = 1;
			Initializer.NumGeometrySegments = 0;
			Initializer.NumCallableShaderSlots = 0;
			Initializer.NumShaderSlotsPerGeometrySegment = 0;
			Initializer.HitGroupIndexingMode = ERayTracingHitGroupIndexingMode::Allow;

			FRHIShaderBindingTable SBT(Initializer);

			FRayTracingShaderBindingsWriter GlobalResources;
			SetShaderParameters(GlobalResources, VActToonRayGenRGS, *PassParameters);

			FRayTracingPipelineStateInitializer PSOInitializer;
			PSOInitializer.MaxPayloadSizeInBytes = GetRayTracingPayloadTypeMaxSize(FVActToonRayGenRGS::GetRayTracingPayloadType(0));;
			PSOInitializer.bAllowHitGroupIndexing = false;

			// Set RayGen shader
			TArray<FRHIRayTracingShader*> RayGenShaderTable;
			RayGenShaderTable.Add(GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FVActToonRayGenRGS>().GetRayTracingShader());
			PSOInitializer.SetRayGenShaderTable(RayGenShaderTable);

			// Set ClosestHit shader
			TArray<FRHIRayTracingShader*> RayHitShaderTable;
			RayHitShaderTable.Add(GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FVActToonRayGenCHS>().GetRayTracingShader());
			PSOInitializer.SetHitGroupTable(RayHitShaderTable);

			// Set Miss shader
			TArray<FRHIRayTracingShader*> RayMissShaderTable;
			RayMissShaderTable.Add(GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FVActToonRayGenMS>().GetRayTracingShader());
			PSOInitializer.SetMissShaderTable(RayMissShaderTable);

			// dispatch ray trace shader
			FRayTracingPipelineState* PipeLine = PipelineStateCache::GetAndOrCreateRayTracingPipelineState(RHICmdList, PSOInitializer);



			RHICmdList.SetRayTracingMissShader(RHIScene, 0, PipeLine, 0 /* ShaderIndexInPipeline */, 0, nullptr, 0);
			RHICmdList.RayTraceDispatch(PipeLine, VActToonRayGenRGS.GetRayTracingShader(), RHIScene, GlobalResources, TextureSize.X, TextureSize.Y);
		}
	);

	// Copy textures from the shader output to our render target
	// this is done as a render pass with the graph builder
	FTextureRHIRef OriginalRT = CachedParams.RenderTarget->GetRenderTargetResource()->GetTexture2DRHI();
	FRDGTexture* OutputRDGTexture = GraphBuilder->RegisterExternalTexture(CreateRenderTarget(ShaderOutputTexture, TEXT("RaytracingToonOutputRT")));
	FRDGTexture* CopyToRDGTexture = GraphBuilder->RegisterExternalTexture(CreateRenderTarget(OriginalRT, TEXT("RaytracingToonCopyToRT")));
	FRHICopyTextureInfo CopyInfo;
	CopyInfo.Size = FIntVector(TextureSize.X, TextureSize.Y, 0);
	AddCopyTexturePass(*GraphBuilder, OutputRDGTexture, CopyToRDGTexture, CopyInfo);
}
#else // !RHI_RAYTRACING
{
	unimplemented();
}
#endif