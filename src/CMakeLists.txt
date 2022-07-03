
# Add source to this project's executable.

file(GLOB SRC_FILES  
    "*.cpp"
    "*.h"
    "*.hpp"
)

add_executable(vulkan_guide
    ${SRC_FILES}
)

set_property(TARGET vulkan_guide PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan_guide>")

target_include_directories(vulkan_guide PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vulkan_guide vkbootstrap vma glm tinyobjloader imgui stb_image tinygltf eigen spherical_harmonics xatlas)

target_link_libraries(vulkan_guide Vulkan::Vulkan sdl2)

add_dependencies(vulkan_guide Shaders)
