cmake_minimum_required(VERSION 3.24)
project(strata_safetensors LANGUAGES CXX)
# This Windows toolchain ships only the Chinese MSVC resource DLL. CMake's
# compiler probe can double-decode its UTF-8 include prefix as GBK. Correct that
# exact detected spelling so Ninja records headers instead of dropping deps.
if(MSVC AND CMAKE_CXX_CL_SHOWINCLUDES_PREFIX MATCHES "娉ㄦ剰")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件: ")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_library(strata_safetensors STATIC
    ${ROOT}/src/weights/safetensors.cpp
    ${ROOT}/src/weights/nvfp4.cpp
    ${ROOT}/src/weights/qwen4.cpp)
target_include_directories(strata_safetensors PUBLIC ${ROOT}/include PRIVATE ${ROOT}/third_party)
add_executable(strata-safetensors ${ROOT}/src/weights/inspect.cpp)
target_link_libraries(strata-safetensors PRIVATE strata_safetensors)
target_include_directories(strata-safetensors PRIVATE ${ROOT}/third_party)
add_executable(safetensors_layout_test ${ROOT}/src/weights/layout_test.cpp)
target_link_libraries(safetensors_layout_test PRIVATE strata_safetensors)
add_executable(safetensors_source_test ${ROOT}/src/weights/source_test.cpp)
target_link_libraries(safetensors_source_test PRIVATE strata_safetensors)
target_include_directories(safetensors_source_test PRIVATE ${ROOT}/third_party)
foreach(target strata_safetensors strata-safetensors safetensors_layout_test safetensors_source_test)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8 /permissive-)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    endif()
endforeach()
add_test(NAME safetensors_layout COMMAND safetensors_layout_test)
add_test(NAME safetensors_source COMMAND safetensors_source_test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME safetensors_reader COMMAND ${Python3_EXECUTABLE}
    ${ROOT}/tools/test_safetensors_reader.py --exe $<TARGET_FILE:strata-safetensors>)
