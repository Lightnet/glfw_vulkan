@echo off
setlocal
set MSYS2_PATH=C:\msys64\ucrt64\bin
set VULKAN_SDK=C:\VulkanSDK\1.4.313.0
set PATH=%MSYS2_PATH%;%VULKAN_SDK%\Bin;%PATH%
if not exist build mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
endlocal