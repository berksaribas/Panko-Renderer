
# Add source to this project's executable.

file(GLOB SRC_FILES  
    "*.cpp"
    "*.h"
    "*.hpp"
    "editor/*.cpp"
    "editor/*.h"
    "editor/*.hpp"
    "memory/*.cpp"
    "memory/*.h"
    "memory/*.hpp"
)

add_executable(panko
    ${SRC_FILES}
)

# set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

set_property(TARGET panko PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:panko>")

target_include_directories(panko PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(panko vkbootstrap vma glm tinyobjloader imgui stb_image tinygltf eigen spherical_harmonics xatlas ffx_fsr2_api_x64 ffx_fsr2_api_vk_x64 fsr2)

target_link_libraries(panko Vulkan::Vulkan sdl2)

add_dependencies(panko Shaders)