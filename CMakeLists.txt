cmake_minimum_required(VERSION 3.14)
project(SimpleTriangle C)

# Explicitly set C standard to avoid C++ modules
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Enable FetchContent
include(FetchContent)

# Declare Vulkan-Headers
FetchContent_Declare(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG vulkan-sdk-1.4.313.0 # Stable tag
    GIT_SHALLOW TRUE
)
# Disable C++ module support
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vulkan_headers)

# Declare Vulkan-Loader
FetchContent_Declare(
    vulkan_loader
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
    GIT_TAG vulkan-sdk-1.4.313.0 # Stable tag
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vulkan_loader)

# Declare GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
# Configure GLFW options
set(GLFW_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS FALSE CACHE BOOL "" FORCE)
set(GLFW_VULKAN_STATIC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Declare cglm
FetchContent_Declare(
    cglm
    GIT_REPOSITORY https://github.com/recp/cglm.git
    GIT_TAG v0.9.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cglm)

set(APP_NAME simple_triangle)

# Create executable
add_executable(${APP_NAME} src/main.c)

# Link libraries
target_link_libraries(${APP_NAME} PRIVATE
    glfw
    vulkan
    cglm
)

# Include directories
target_include_directories(${APP_NAME} PRIVATE
    ${vulkan_headers_SOURCE_DIR}/include
    ${cglm_SOURCE_DIR}/include
)

# Shader compilation with glslc from Vulkan SDK
set(VULKAN_SDK "C:/VulkanSDK/1.4.313.0")
set(GLSLC "${VULKAN_SDK}/Bin/glslc.exe")
set(SHADER_SRC
    ${CMAKE_SOURCE_DIR}/shaders/shader.vert
    ${CMAKE_SOURCE_DIR}/shaders/shader.frag
)
foreach(SHADER ${SHADER_SRC})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SHADER_SPV "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC} ${SHADER} -o ${SHADER_SPV}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    list(APPEND SHADER_SPV_FILES ${SHADER_SPV})
endforeach()
add_custom_target(shaders DEPENDS ${SHADER_SPV_FILES})
add_dependencies(${APP_NAME} shaders)

# Optional: Enable stricter compiler warnings
target_compile_options(${APP_NAME} PRIVATE -Wall -Wextra)

# Copy shaders to build directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
file(COPY ${CMAKE_SOURCE_DIR}/shaders/ DESTINATION ${CMAKE_BINARY_DIR}/shaders)