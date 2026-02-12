# VActUEPlugins
Vivacity Interactive UE Plugins, generally copy the plugin in context to your Projects ``Plugins`` folder or Engine ``Plugins`` folder.

![Static Badge](https://img.shields.io/badge/Tip-green)
_Adviced way to ``clone`` only particular plugins use  ``sparse-checkout``, see example below for ``main`` branch_
```git-bash
git clone --no-checkout https://github.com/Vivacity-Interactive/VActUEPlugins.git Plugins
cd Plugins
git sparse-checkout init
git sparse-checkout set VActBase VActOIC VActFiles
git checkout main
```
![Static Badge](https://img.shields.io/badge/Caution-red)
_Don't forget ``cd Plugins`` before setting ``sparse-checkout``! If you forget: current git repro will be compromised_

List of Plugins
- [VActBase](VActBase/README.md)
- [VActFiles](VActFiles/README.md)
- [VActOIC](VActOIC/README.md) _(beta, oic.v3, vson.v2)_
- [VActDevices](VActDevices/README.md) _(alpha)_
- [VActCuDNN](VActCuDNN/README.md) _(Unfinished)_
- [VActML](VActML/README.md) _(boiler)_
- [VActVR](VActVR/README.md)
- [VActCamera](VActCamera/README.md)
- [VActMath](VActMath/README.md) _(alpha)_
- [VActSTT](VActSTT/README.md) _(beta)_
- [VActAPI](VActAPI/README.md) _(dev-limited)_
- [VActBeats](VActAPI/README.md) _(wip, [dev-api-1.0.0](https://github.com/Vivacity-Interactive/VActUEPlugins/tree/dev-api-1.0.0))_

## VActBase
Contains the basic library items Vivacity Interactive uses for most projects as bases, includes, also wrapper functinos TArray manipulations. [VActBase](VActBase/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_still working out what to put here, will adress a refactor to make the use of ``TStackHandle<T>`` more usable_

## VActFiles
This plugin allows for loading VAct Json (Vson), and has other helper files for Tokenizing and reading and interpreting files. [VActFiles](VActFiles/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_My be refactored to make the use of Tokenization and Emit more easy_

## VActOIC
This plugin allows for loading VAct Json (Vson) type files and instantiate entire compositions of objects like meshes, and particles, also includes construction ``SceneComponent`` or ``ActorComponent`` onto object that have these meta data assigned. [VActOIC](VActOIC/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_tested on 200.000+ instances, now also includes a limited exporter for UE_

![Static Badge](https://img.shields.io/badge/Important-purple)
_This plugin depends on the ``VActFiles`` plugin, formats compact and binary ar not yet supported_

![Static Badge](https://img.shields.io/badge/Warning-yellow)
_Update breaks using tracking, sor reloading atm requires you to delete the OICActor and reassign and update again_

## VActDevices
Allows to communicate with COM devices and Bluetooth, tested with Arduino simple setup. [VActDevices](VActDevices/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_Bluetooth is still incomplete, need to setup connection protocol and test, listing available devices is already successful, also still resolving best usage approach_

## VActCuDNN
This is still work in progress, attempt to bring more native ML to unreal engine, but this may cause GPU resource competition. best may be to hook into UE intial support with Cuda and CuDNN duable. [VActCuDNN](VActCuDNN/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_similar attempt using tensorflow direct support, however tensorflow core packag had some compilation flag issues, and tensorflow full library depends on pythorgh and python compilation, this last dependecy I prefer to avoid_

## VActML
This is still work in progress, attempt to bring more native ML to unreal engine, independed of too many external frameworks and packages for runtime use. earliest focuses will be by order. [VActML](VActML/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_will also attempt to focus on including compute shader versions of algorithms, maybe tensor shader if internally accessible by unreal engine, or abusing other shaders that may be isomorphic in behaviour_

## VActVR
VR C++ implementation, and Contains default blueprint derivatives, containing interaction components and other VR specific helpers. Specifically focused on keeping clear separation from VR specific `Pawn` as Camera and Embodied representation. [VActVR](VActVR/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_interaction components are still being implemented, and may change_

## VActCamera
Noir-Like Camera, as if an operator is handling it, allows usage of hints and samplers to match camera behaviours to areas. Hints contain info about camera settings and behaviour plus suggestive location and rotation. [VActCamera](VActCamera/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_Still working a minimal version and a complex versions, and may merge it with a dependency on ``VActMath``_

## VActMath
ALl math functions for pointer type arrays, to use with `reinterpret_cast<float*>(&MyFloatPropertyStruct)` or just (float*)&MyFloatPropertyStruct. Functions like `Dot`, `Lerp`, `Clamp` and more. [VActMath](VActMath/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_it may be extended with tenrsor like behaviours like poll and kernel like functions for multidimensions_

## VActSTT
Functions for speach to text using a compilation of wisper cpp library, wrapped into UE like classes usable in Games, loading models localy. [VActSTT](VActSTT/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_will be extended with more functions part of wisper cpp as well_

## VActAPI (See Branch dev-api-1.0.0)
Allows for running server like behaviour (an aip-route-service) on your game instance on a local network, allows for direction connection like uploading images or other assets, or simply to communcate, allowing `Blueprint` extension on callbacks. [VActAPI](VActAPI/README.md)

![Static Badge](https://img.shields.io/badge/Warning-yellow)
_Unfortunately, still, due to the strange nature of its singleton behaviour, the API does not work well with PIE, so testing can only work at first startup, a re run of PIE or compilation will break the references_

## VActBeats (wip)
Features a particular approach of story event management, rooted from emotional state spaces allowing, using techniques like `KDTree` for `BeatVectors` sampling from `BeatPool`s or (linear) `BeatStack`s. [VActBeats](VActBeats/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_stil under construction_