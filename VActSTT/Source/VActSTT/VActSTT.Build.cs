using System.IO;

using UnrealBuildTool;

public class VActSTT : ModuleRules
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
	
	public VActSTT(ReadOnlyTargetRules Target) : base(Target)
	{

		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicAdditionalLibraries.AddRange(
			new string[] {
                Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "WhisperCpp", "whisper", true)),
                Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "GGML", "ggml-cpu", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "GGML", "ggml-base", true)),
				Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "GGML", "ggml", true)),
			}
			);

		// RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "WhisperCpp", "whisper", false)));
		// RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "GGML", "ggml-cpu", false)));
		// RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "GGML", "ggml-base", false)));
		// RuntimeDependencies.Add(Path.Combine(ModuleDirectory, "..", LibraryPath(Target, "GGML", "ggml", false)));

		// PublicDelayLoadDLLs.AddRange(
		// 	new string[] {
		// 		LibraryFile(Target, "whisper", false),
		// 		LibraryFile(Target, "ggml-cpu", false),
		// 		LibraryFile(Target, "ggml-base", false),
		// 		LibraryFile(Target, "ggml", false),
		// 	}
		// 	);
		
		PublicIncludePaths.AddRange(
			new string[] {
               Path.Combine(ModuleDirectory, "../ThirdParty/WhisperCpp/Public"),
				// ... add public include paths required here ...
			}
			);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory, "../ThirdParty/GGML/Public"),
				Path.Combine(ModuleDirectory, "../ThirdParty/WhisperCpp/Public"),
			}
			);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
                "AudioCapture",
                "AudioCaptureCore",
                "AudioPlatformConfiguration",
                "AudioMixer",
                "AudioMixerCore",
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
