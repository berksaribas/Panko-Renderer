cmake_minimum_required (VERSION 3.8)

project ("panko")

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(src)


find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

## find all the shader files under the shaders folder
file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    "${PROJECT_SOURCE_DIR}/shaders/*.rgen"
    "${PROJECT_SOURCE_DIR}/shaders/*.rmiss"
    "${PROJECT_SOURCE_DIR}/shaders/*.rchit"
    )
file(GLOB_RECURSE GLSL_HEADER_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.glsl"
    )

## iterate each shader
foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  get_filename_component(FILE_DIRECTORY ${GLSL} DIRECTORY)
  set(SPIRV "${FILE_DIRECTORY}/${FILE_NAME}.spv")
  message(STATUS ${GLSL})
  ##execute glslang command to compile that specific shader
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} --target-env spirv1.4 -V ${GLSL} -o ${SPIRV} -g
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )

target_sources(Shaders PRIVATE 
    ${GLSL_SOURCE_FILES}
    ${GLSL_HEADER_FILES}
    )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ FILES ${GLSL_SOURCE_FILES} )
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ FILES ${GLSL_HEADER_FILES} )
message(${GLSL_SOURCE_FILES})
