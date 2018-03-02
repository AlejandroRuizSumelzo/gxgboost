cmake_minimum_required(VERSION 3.2)

project(gxgboost CXX)

include(GenerateExportHeader)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)

add_subdirectory(xgboost)

include_directories (
    xgboost/src
    xgboost/include
    xgboost/dmlc-core/include
    xgboost/rabit/include
)

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
if(MSVC)
  # Multithreaded compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  # Correct error for GCC 5 and cuda
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
  # Performance
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unknown-pragmas")

add_library(gxgboost SHARED src/gxgboost.cc)
target_compile_definitions(gxgboost PRIVATE gxgboost_EXPORTS)
target_include_directories(gxgboost PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include/gxgboost>)
target_link_libraries(gxgboost PRIVATE xgboost)

generate_export_header(gxgboost BASE_NAME GXGBOOST)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gxgboost_export.h src/gxgboost.h DESTINATION include/gxgboost)
install(TARGETS gxgboost EXPORT GXGBoostConfig DESTINATION lib)
install(EXPORT GXGBoostConfig DESTINATION cmake)

