using System.IO;

using UnrealBuildTool;

public class VActCuDNN : ModuleRules
{
	private string LibraryFile(ReadOnlyTargetRules Target, string FileName, bool bStatic = false)
	{
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string Ext = bStatic ? "lib" : "dll";
			return $"{FileName}.{Ext}";
		}
		
		if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			string Ext = bStatic ? "a" : "so";
			return $"{FileName}.{Ext}";
		}

		if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			string Ext = bStatic ? "a" : "dylib";
			return $"{FileName}.{Ext}";
		}
		return null;
	}

	private string LibraryPath(ReadOnlyTargetRules Target, string LibName, string FileName, bool bStatic = false)
	{
		string BaseDir = Path.Combine("ThirdParty", LibName);

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			return Path.Combine(BaseDir, "Win64", LibraryFile(Target, FileName, bStatic));
		}
		
		if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			return Path.Combine(BaseDir, "Linux", LibraryFile(Target, FileName, bStatic));
		}

		if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			return Path.Combine(BaseDir, "Mac", LibraryFile(Target, FileName, bStatic));
		}
		return null;
	}

	public VActCuDNN(ReadOnlyTargetRules Target) : base(Target)
	{

		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicAdditionalLibraries.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "Cuda", "cuda", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "Cuda", "cudart", true)),

				// //Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn64_8", true)),
				//Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn_ops_infer", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_ops_infer64_8", true)),
				//Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn_ops_train", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_ops_train64_8", true)),
				//Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn_cnn_infer", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_cnn_infer64_8", true)),
				//Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn_cnn_train", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_cnn_train64_8", true)),
				// //Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn_adv_infer", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_adv_infer64_8", true)),
				// //Path.Combine(ModuleDirectory, LibraryPath(Target, "CuDNN", "cudnn_adv_train", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_adv_train64_8", true)),
			}
			);

		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "Cuda", "cudart64_12", false)));

		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn64_8", false)));
		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_ops_infer64_8", false)));
		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_ops_train64_8", false)));
		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_cnn_infer64_8", false)));
		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_cnn_train64_8", false)));
		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_adv_infer64_8", false)));
		RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "CuDNN", "cudnn_adv_train64_8", false)));

		PublicDelayLoadDLLs.AddRange(
			new string[] {
				LibraryFile(Target, "cudart64_12", false),

				LibraryFile(Target, "cudnn64_8", false),
				LibraryFile(Target, "cudnn_ops_infer64_8", false),
				LibraryFile(Target, "cudnn_ops_train64_8", false),
				LibraryFile(Target, "cudnn_cnn_infer64_8", false),
				LibraryFile(Target, "cudnn_cnn_train64_8", false),
				LibraryFile(Target, "cudnn_adv_infer64_8", false),
				LibraryFile(Target, "cudnn_adv_train64_8", false),
			}
			);

		PublicIncludePaths.AddRange(
			new string[] {
				// ... add public include paths required here ...
			}
			);
		
		PrivateIncludePaths.AddRange(
			new string[] {
				"ThirdParty/Cuda/Public",
				"ThirdParty/CuDNN/Public"
				// ... add other private include paths required here ...
			}
			);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"Projects"
				// ... add other public dependencies that you statically link with here ...
			}
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
			);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);
	}
}
