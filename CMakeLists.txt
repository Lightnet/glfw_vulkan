cmake_minimum_required(VERSION 3.14)
project(SimpleTriangle C)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Enable FetchContent
include(FetchContent)

# Declare GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

# Configure GLFW options
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_VULKAN_STATIC OFF CACHE BOOL "" FORCE)

# Declare cglm
FetchContent_Declare(
    cglm
    GIT_REPOSITORY https://github.com/recp/cglm.git
    GIT_TAG v0.9.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cglm)

set(APP_NAME simple_triangle)

# Create executable
add_executable(${APP_NAME} src/main.c)

# Link libraries
target_link_libraries(${APP_NAME} PRIVATE
    glfw
    Vulkan::Vulkan
    cglm
)

# Include directories
target_include_directories(${APP_NAME} PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${cglm_SOURCE_DIR}/include
)

# Shader compilation
set(SHADER_SRC
    ${CMAKE_SOURCE_DIR}/shaders/shader.vert
    ${CMAKE_SOURCE_DIR}/shaders/shader.frag
)
foreach(SHADER ${SHADER_SRC})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SHADER_SPV "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER} -o ${SHADER_SPV}
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