set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
set(EIGEN3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen/cmake")
SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/eigen/cmake")

find_package(Eigen3 3.4.0 REQUIRED)

find_package(Vulkan REQUIRED)

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)

add_library(tinygltf INTERFACE)

add_library(tinyobjloader STATIC)

add_library(stb_image INTERFACE)

add_library(spherical_harmonics STATIC)

add_library(eigen INTERFACE)

add_library(xatlas STATIC)

target_sources(vkbootstrap PRIVATE 
    vkbootstrap/VkBootstrap.h
    vkbootstrap/VkBootstrap.cpp
    )

file(GLOB SRC_FILES_SPHERICAL  
    "spherical_harmonics/*.cc"
    "spherical_harmonics/*.h"
)

target_sources(spherical_harmonics PRIVATE 
    ${SRC_FILES_SPHERICAL}
)

target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)

#both vma and glm and header only libs so we only need the include path
target_include_directories(vma INTERFACE vma)
target_include_directories(glm INTERFACE glm)
target_include_directories(tinygltf INTERFACE tinygltf)
target_include_directories(eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

target_include_directories(spherical_harmonics PUBLIC spherical_harmonics)
target_link_libraries(spherical_harmonics PUBLIC eigen $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)

target_sources(tinyobjloader PRIVATE 
    tinyobjloader/tiny_obj_loader.h
    tinyobjloader/tiny_obj_loader.cc
    )

target_include_directories(tinyobjloader PUBLIC tinyobjloader)


target_sources(xatlas PRIVATE 
    xatlas/xatlas.cpp
    xatlas/xatlas.h
    )

target_include_directories(xatlas PUBLIC xatlas)

add_library(sdl2 INTERFACE)
set(sdl2_DIR "SDL_PATH" CACHE PATH "Path to SDL2")

if (WIN32)
target_include_directories(sdl2 INTERFACE ${sdl2_DIR}/include ${sdl2_image_DIR}/include)
target_link_directories(sdl2 INTERFACE ${sdl2_DIR}/lib/x64 ${sdl2_image_DIR}/lib/x64)
elseif (UNIX)
target_include_directories(sdl2 INTERFACE ${sdl2_DIR}/include/SDL2 ${sdl2_image_DIR}/include/SDL2)
target_link_directories(sdl2 INTERFACE ${sdl2_DIR}/lib ${sdl2_image_DIR}/lib)
endif()

target_link_libraries(sdl2 INTERFACE SDL2 SDL2main)

add_library(imgui STATIC)

target_include_directories(imgui PUBLIC imgui)

target_sources(imgui PRIVATE 
    imgui/imgui.h
    imgui/imgui.cpp

    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp

    imgui/imgui_impl_vulkan.cpp
    imgui/imgui_impl_sdl.cpp
    )

target_link_libraries(imgui PUBLIC Vulkan::Vulkan sdl2)

target_include_directories(stb_image INTERFACE stb_image)