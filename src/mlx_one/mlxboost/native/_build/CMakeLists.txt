cmake_minimum_required(VERSION 3.27)
project(mlxboost_native LANGUAGES CXX OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/../../../mlx/lib")
set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../../../mlx/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import importlib.util; print(next(iter(importlib.util.find_spec('mlx').submodule_search_locations)) + '/share/cmake/MLX')"
    OUTPUT_VARIABLE MLX_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(MLX 0.32.2 EXACT CONFIG REQUIRED PATHS "${MLX_ROOT}" NO_DEFAULT_PATH)
if(NOT APPLE OR NOT MLX_BUILD_METAL)
    message(FATAL_ERROR "MLXBoost requires the pinned Apple Metal MLX build")
endif()
file(GLOB identity_sources CONFIGURE_DEPENDS "*.cpp" "*.mm" "*.h" "*.py" "*.metal" "*.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${identity_sources})
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/build_identity.py"
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_build_info.json"
    OUTPUT_VARIABLE BUILD_IDENTITY OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build_identity.h" "${BUILD_IDENTITY}")
add_library(mlxboost_ops SHARED axpby.cpp gemv.cpp)
target_include_directories(mlxboost_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(mlxboost_ops PRIVATE -fno-fast-math -ffp-contract=off)
target_link_libraries(mlxboost_ops PUBLIC mlx)
set_target_properties(mlxboost_ops PROPERTIES INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_NAME_DIR ON)
mlx_build_metallib(TARGET mlxboost_metallib TITLE mlxboost
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/axpby.metal" "${CMAKE_CURRENT_SOURCE_DIR}/gemv.metal"
    INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
add_dependencies(mlxboost_ops mlxboost_metallib)
nanobind_add_module(_ext NB_STATIC NB_DOMAIN mlx bindings.cpp replay.mm replay_recorder.mm replay_reflection.mm)
target_compile_options(_ext PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_include_directories(_ext PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(_ext PRIVATE mlxboost_ops "-framework Metal" "-framework Foundation")
