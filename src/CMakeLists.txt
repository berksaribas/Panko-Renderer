
# Add source to this project's executable.
add_executable(vulkan_guide
    main.cpp
    vk_engine.cpp
    vk_engine.h
    vk_types.h
    vk_initializers.cpp
    vk_initializers.h
    vk_pipeline.cpp
    vk_pipeline.h
    vk_mesh.cpp
    vk_mesh.h
    vk_texture.cpp
    vk_texture.h
    vk_shader.cpp
    vk_shader.h
    gltf_scene.cpp
    gltf_scene.hpp
    precalculation.cpp
    precalculation.h
    triangle_box_intersection.h)


set_property(TARGET vulkan_guide PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan_guide>")

target_include_directories(vulkan_guide PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vulkan_guide vkbootstrap vma glm tinyobjloader imgui stb_image tinygltf optick)

target_link_libraries(vulkan_guide Vulkan::Vulkan sdl2)

add_dependencies(vulkan_guide Shaders)
