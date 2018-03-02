cmake_minimum_required(VERSION 3.2)

project(gxgboost VERSION 0.1.0 LANGUAGES CXX)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")

add_library(gxgboost SHARED src/gxgboost.cc)
target_compile_definitions(gxgboost PRIVATE gxgboost_EXPORTS)
target_include_directories(gxgboost PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include/gxgboost>)
target_link_libraries(gxgboost PRIVATE xgboost)

generate_export_header(gxgboost BASE_NAME GXGBOOST)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")

function(INSTALL_PACKAGE_FILES FILES_LIST SRC_DIR DEST_DIR PACKAGE_FILES_VAR)
    foreach(SRC_FILE IN LISTS FILES_LIST)
        message(STATUS "Doing install step for file ${SRC_FILE}")
        install(FILES ${SRC_DIR}/${SRC_FILE} DESTINATION ${DEST_DIR})
        set(${PACKAGE_FILES_VAR} "${${PACKAGE_FILES_VAR}} ${DEST_DIR}/${SRC_FILE}")
    endforeach()
    set(${PACKAGE_FILES_VAR} "${${PACKAGE_FILES_VAR}}" PARENT_SCOPE)
endfunction()

install(TARGETS gxgboost DESTINATION dlib)

set(PACKAGE_FILES )
install_package_files("gxgboost.sdf;gxgboost.src" data src PACKAGE_FILES)
install_package_files("gxgboost.e;agaricus-lepiota.csv" data examples PACKAGE_FILES)
install_package_files("package.json" res "${CMAKE_INSTALL_PREFIX}" PACKAGE_FILES)
string(REPLACE "${CMAKE_INSTALL_PREFIX}/" "" PACKAGE_FILES "${PACKAGE_FILES}")

set(GXGBOOST_GENERATED_FILE ${CMAKE_CURRENT_BINARY_DIR}/gxgboost_output_name.cmake)
file(GENERATE 
    OUTPUT ${GXGBOOST_GENERATED_FILE}
    CONTENT "
    execute_process(COMMAND \"${CMAKE_COMMAND}\" -E tar cvf \"${CMAKE_CURRENT_BINARY_DIR}/gxgboost.zip\" --format=zip ${PACKAGE_FILES} dlib/$<TARGET_FILE_NAME:gxgboost> WORKING_DIRECTORY \"${CMAKE_INSTALL_PREFIX}\")
    "
    CONDITION 1
)

install(CODE "include(${GXGBOOST_GENERATED_FILE})")

