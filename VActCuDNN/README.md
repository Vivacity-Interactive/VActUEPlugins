# VActCuDNN
This is still work in progress, attempt to bring more native ML to unreal engine, but this may cause GPU resource competition. best may be to hook into UE intial support with Cuda and CuDNN duable. [VActCuDNN](VActCuDNN/README.md)

![Static Badge](https://img.shields.io/badge/Note-blue)
_similar attempt using tensorflow direct support, however tensorflow core packag had some compilation flag issues, and tensorflow full library depends on pythorgh and python compilation, this last dependecy I prefer to avoid_

## Install
Download ``ThirdParty`` Source Folder. Unpack the downloaded third party folder into ``VActCuDNN/Source/``. It contains Cuda and CuDNN dependencies.

![Static Badge](https://img.shields.io/badge/Note-blue)
_I may rewrite it to depend on the general SDK installation of Cuda and CuDNN, depending on Path settings and System Variables, this current take is an attempt to make it somewhat stand alone_

- [VActCuDNN_Source_ThirdParty.zip](https://drive.google.com/file/d/1gFahD7kSydta6d4YZs3LgYqo54lmAFfP/view?usp=sharing "ThirdParty zip file dependencies")