#include "VActCuDNNModule.h"

#include "Misc/MessageDialog.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

#if WITH_EDITOR
#include "_VActCuDNN_Tests.h"
#endif;

// #include "ThirdParty/Cuda/Public/Cuda/cuda_runtime.h"
// #include "ThirdParty/cudnn/Public/cudnn/cudnn.h"

#define LOCTEXT_NAMESPACE "FVActCuDNNModule"

#define _VACT_LOADLIB(LibOut, LibPathIn, LibNameIn)											\
{																							\
	LibOut = !LibPathIn.IsEmpty() ? FPlatformProcess::GetDllHandle(*LibPathIn) : nullptr;	\
	bool bFail = !(LibOut);																	\
	if (bFail)																				\
	{																						\
		UE_LOG(LogTemp, Fatal, TEXT("VActCuDNN \"%s\" failed to load \"%s\"")					\
			, LibNameIn, *LibPathIn);														\
	}																						\
	else																					\
	{																						\
		UE_LOG(LogTemp, Display, TEXT("VActCuDNN \"%s\" successfully loaded \"%s\"")			\
			, LibNameIn, *LibPathIn);														\
	}																						\
}

#define _VACT_FREELIB(LibIn)																\
{																							\
	FPlatformProcess::FreeDllHandle(LibIn);													\
	LibIn = nullptr;																		\
}																							\

void FVActCuDNNModule::StartupModule()
{
	FString BaseDir = IPluginManager::Get().FindPlugin("VActCuDNN")->GetBaseDir();

	FString CudaLibPath;

	FString CuDNNLibPath;
	FString CuDNNOpsInferLibPath;
	FString CuDNNOpsTrainLibPath;
	FString CuDNNCNNInferLibPath;
	FString CuDNNCNNTrainLibPath;
	FString CuDNNAdvInferLibPath;
	FString CuDNNAdvTrainLibPath;

	// TODO maybe support custom property referencing System Library Director CUDA and CuDNN
#if PLATFORM_WINDOWS
	CudaLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/Cuda/Win64/cudart64_12.dll"));

	CuDNNLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn64_8.dll"));
	CuDNNOpsInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_ops_infer64_8.dll"));
	CuDNNOpsTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_ops_train64_8.dll"));
	CuDNNCNNInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_cnn_infer64_8.dll"));
	CuDNNCNNTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_cnn_train64_8.dll"));
	CuDNNAdvInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_adv_infer64_8.dll"));
	CuDNNAdvTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_adv_train64_8.dll"));
#elif PLATFORM_MAC
	CudaLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/Cuda/Mac/cudart64_12.dylib"));

	CuDNNLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn64_8.dylib"));
	CuDNNOpsInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_ops_infer64_8.dylib"));
	CuDNNOpsTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_ops_train64_8.dylib"));
	CuDNNCNNInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_cnn_infer64_8.dylib"));
	CuDNNCNNTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_cnn_train64_8.dylib"));
	CuDNNAdvInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_adv_infer64_8.dylib"));
	CuDNNAdvTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_adv_train64_8.dylib"));
#elif PLATFORM_LINUX
	CudaLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/Cuda/Linux/cudart64_12.so"));

	CuDNNLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn64_8.so"));
	CuDNNOpsInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_ops_infer64_8.so"));
	CuDNNOpsTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_ops_train64_8.so"));
	CuDNNCNNInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_cnn_infer64_8.so"));
	CuDNNCNNTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_cnn_train64_8.so"));
	CuDNNAdvInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_adv_infer64_8.so"));
	CuDNNAdvTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_adv_train64_8.so"));
#endif

	_VACT_LOADLIB(CudaLib, CudaLibPath, TEXT("Cuda"));

	_VACT_LOADLIB(CuDNNLib, CuDNNLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNOpsInferLib, CuDNNOpsInferLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNOpsTrainLib, CuDNNOpsTrainLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNCNNInferLib, CuDNNCNNInferLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNCNNTrainLib, CuDNNCNNTrainLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNAdvInferLib, CuDNNAdvInferLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNAdvTrainLib, CuDNNAdvTrainLibPath, TEXT("CuDNN"));

#if WITH_EDITORONLY_DATA
	//_FVActCuDNNTests::_Debug_Cuda_Test();
	//_FVActCuDNNTests::_Debug_CuDNN_Test_0();
	//_FVActCuDNNTests::_Debug_CuDNN_Test_1();
	//_FVActCuDNNTests::_Debug_VActCuDNN_Test_0();
	//_FVActCuDNNTests::_Debug_VActCuDNN_Test_1();
#endif;
}

void FVActCuDNNModule::ShutdownModule()
{
	_VACT_FREELIB(CudaLib);

	_VACT_FREELIB(CuDNNLib);
	_VACT_FREELIB(CuDNNOpsTrainLib);
	_VACT_FREELIB(CuDNNOpsInferLib);
	_VACT_FREELIB(CuDNNCNNInferLib);
	_VACT_FREELIB(CuDNNCNNTrainLib);
	_VACT_FREELIB(CuDNNAdvTrainLib);
	_VACT_FREELIB(CuDNNAdvInferLib);
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActCuDNNModule, VActCuDNN)