// Copyright Vivaicty Interactive, Inc. All Rights Reserved.

#pragma once

#include "Modules/ModuleManager.h"

class FVActCuDNNModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	/** Handle to the test dll we will load */
	void* CudaLib;

	void* CuDNNLib;
	void* CuDNNOpsTrainLib;
	void* CuDNNOpsInferLib;
	void* CuDNNCNNTrainLib;
	void* CuDNNCNNInferLib;
	void* CuDNNAdvInferLib;
	void* CuDNNAdvTrainLib;
};

#if WITH_EDITOR 
struct FVActCuDNNTests
{
	static void _Debug_Cuda_Test();

	static void _Debug_CuDNN_Test_0();

	static void _Debug_CuDNN_Test_1();

	static void _Debug_VActCuDNN_Test_0();

	static void _Debug_VActCuDNN_Test_1();
};
#endif;