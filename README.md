# glfw_vulkan

# License: MIT

# Informtion:
    Using the msys2 to test build glfw and vulkan. Sample test for cmake build.

# Prerequisites
- msys2-x86_64-xxx.exe
- python ?
- 

```
pacman -S package_name
```
note this refs package if missing. Testing vulkan and opengl.
```
mingw-w64-ucrt-x86_64-gcc
mingw-w64-ucrt-x86_64-cmake 
mingw-w64-ucrt-x86_64-ninja 
mingw-w64-ucrt-x86_64-git
```
```
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-git
```
- Need correct package to install test.
- To keep the tool compiler size down.


```
project_root/
├── build.bat
├── CMakeLists.txt
├── src/
│   └── main.c
├── shaders/
│   ├── shader.vert
│   └── shader.frag
├── build/ (generated during build)
│   └── shaders/
│       ├── shader.vert.spv
│       └── shader.frag.spv
```

