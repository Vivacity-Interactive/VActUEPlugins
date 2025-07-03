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
- [VActBase](#vactbase)
- [VActFiles](#vactfiles) _(Beta)_
- [VActOIC](#vactoic) _(Alpha)_
- [VActDevices](#vactdevices) _(Alpha)_
- [VActCuDNN](#vactcudnn) _(Unfinished)_
- [VActML](#vactml) _(boiler)_
- [VActVR](#vactvr)

## VActBase
Contains the basic library items Vivacity Interactive uses for most projects as bases, includes
- Default Function Library (Math, Casting, Construction, Filtering)
- Handlers for ``TArrays`` (``List``, ``Stack``, ``Queue``, ``Iterate``)
- Random sampler ``Sampler``
- Event Queue Based ``NPCControler`` base ``AIControler`` Actor
- Default Character ``VActCharacter`` Actor
- Blender equivalent `Empty` Actor

## VActFiles
This plugin allows for loading VAct Json (Vson), and has other helper files for Tokenizing and reading and interpreting files.

## VActOIC
This plugin allows for loading VAct Json (Vson) type files and instantiate entire compositions of objects like meshes, and particles, also includes construction ``SceneComponent`` or ``ActorComponent`` onto object that have these meta data assigned.

![Static Badge](https://img.shields.io/badge/Important-purple)
_This plugin depends on the ``VActFiles`` plugin, formats compact and binary ar not yet supported_

### Usage
- Create a ``OICProfile`` (Blueprint), reference it to a file
- Drag a ``OICActor`` into the scene, select ``OICComponent`` and assign a created ``OICProfile`` and press update
- Profile instantiation can also be managed in ``OICManagerActor``

![Static Badge](https://img.shields.io/badge/Warning-yellow)
_Update breaks using tracking, sor reloading atm requires you to delete the OICActor and reassign and update again_

## VActDevices
Allows to communicate with COM devices and Bluetooth, tested with Arduino simple setup.
- COM Devices
- Bluetooth _(WIP)_

![Static Badge](https://img.shields.io/badge/Note-blue)
_Bluetooth is still incomplete, need to setup connection protocol and test, listing available devices is already successful, also still resolving best usage approach_

## VActCuDNN
This is still work in progress, attempt to bring more native ML to unreal engine, but this may cause GPU resource competition. best may be to hook into UE intial support with Cuda and CuDNN duable.

![Static Badge](https://img.shields.io/badge/Note-blue)
_similar attempt using tensorflow direct support, however tensorflow core packag had some compilation flag issues, and tensorflow full library depends on pythorgh and python compilation, this last dependecy I prefer to avoid_

## VActML
This is still work in progress, attempt to bring more native ML to unreal engine, independed of too many external frameworks and packages for runtime use. earliest focuses will be by order.
- Proximal Policy Optimization (PPO)
- Proximal Policy Optimization 2 (PPO2)
- Continuous K-Nearest Neighbour (CKNN) 
- Large Language Model (LLM) Minimal Base Implementation

![Static Badge](https://img.shields.io/badge/Note-blue)
_will also attempt to focus on including compute shader versions of algorithms, maybe tensor shader if internally accessible by unreal engine, or abusing other shaders that may be isomorphic in behaviour_

### Install
Download ``ThirdParty`` Source Folder. Unpack the downloaded third party folder into ``VActCuDNN/Source/``. It contains Cuda and CuDNN dependencies.

![Static Badge](https://img.shields.io/badge/Note-blue)
_I may rewrite it to depend on the general SDK installation of Cuda and CuDNN, depending on Path settings and System Variables, this current take is an attempt to make it somewhat stand alone_

- [VActCuDNN_Source_ThirdParty.zip](https://drive.google.com/file/d/1gFahD7kSydta6d4YZs3LgYqo54lmAFfP/view?usp=sharing "ThirdParty zip file dependencies")

## VActVR
VR C++ implementation, and Contains default blueprint derivatives 
- ``BP_VRCameraPawn``
- ``BP_VRPlayerControler``
- ``BP_VRGameMode``