# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#
# arrow_metal
#
# Metal-coherent buffer / device / memory-manager backend for Apache Arrow.
# Apple Silicon has unified memory, so a single MTLBuffer allocated with
# MTLResourceStorageModeShared is simultaneously CPU-accessible (via
# .contents) and GPU-accessible (as MTLBuffer). This library exposes that
# duality through arrow::Buffer / arrow::Device / arrow::MemoryManager
# subclasses without ever copying data between host and device.

add_custom_target(arrow_metal-all)
add_custom_target(arrow_metal)
add_custom_target(arrow_metal-benchmarks)
add_custom_target(arrow_metal-tests)
add_dependencies(arrow_metal-all arrow_metal arrow_metal-tests
                 arrow_metal-benchmarks)

# Apple-platform + OBJCXX guards live in the top-level CMakeLists.txt
# under `if(ARROW_METAL)` (CMAKE_OBJCXX_COMPILE_OBJECT must be set before
# this directory is descended into).

# Locate the SDK frameworks we link against. These ship with the macOS
# SDK so they are always available, but using find_library exposes the
# absolute path to downstream consumers via the exported target.
find_library(ARROW_METAL_FRAMEWORK Metal REQUIRED)
find_library(ARROW_FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Capture the SDK version so metal_version.h reflects the build environment.
# CMake exposes the SDK in CMAKE_OSX_SYSROOT; parse the SDKSettings.plist
# for the version. Fall back to 0.0 if anything goes wrong (header is
# advisory, not load-bearing).
set(ARROW_METAL_SDK_VERSION_MAJOR 0)
set(ARROW_METAL_SDK_VERSION_MINOR 0)
if(CMAKE_OSX_SYSROOT)
  execute_process(COMMAND xcrun --sdk macosx --show-sdk-version
                  OUTPUT_VARIABLE _arrow_metal_sdk_version
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(_arrow_metal_sdk_version MATCHES "^([0-9]+)\\.([0-9]+)")
    set(ARROW_METAL_SDK_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(ARROW_METAL_SDK_VERSION_MINOR ${CMAKE_MATCH_2})
  endif()
endif()

# pkg-config private cflags. Static-link consumers need ARROW_METAL_STATIC.
if(NOT ARROW_BUILD_SHARED AND ARROW_BUILD_STATIC)
  string(APPEND ARROW_METAL_PC_CFLAGS "${ARROW_METAL_PC_CFLAGS_PRIVATE}")
  set(ARROW_METAL_PC_CFLAGS_PRIVATE "")
endif()

# Sources. Each task in the metal-native plan appends to this list as it
# lands its files. Ordering does not matter for the link.
set(ARROW_METAL_SRCS metal_init.cc
                     metal_buffer.mm
                     metal_device.mm
                     metal_memory.mm
                     metal_memory_pool.mm)

# .mm sources should compile with ARC; the bridge between Apple's reference
# counting and arrow's std::shared_ptr lifetimes assumes ARC ownership.
foreach(_src IN LISTS ARROW_METAL_SRCS)
  if(_src MATCHES "\\.mm$")
    set_source_files_properties(${_src}
                                PROPERTIES COMPILE_FLAGS "-fobjc-arc"
                                           SKIP_PRECOMPILE_HEADERS ON)
  endif()
endforeach()

set(ARROW_METAL_LINK_LIBS arrow::flatbuffers)
set(ARROW_METAL_SHARED_LINK_LIBS ${ARROW_METAL_FRAMEWORK}
                                 ${ARROW_FOUNDATION_FRAMEWORK})

set(ARROW_METAL_PKG_CONFIG_NAME_ARGS)
if(NOT WINDOWS)
  list(APPEND ARROW_METAL_PKG_CONFIG_NAME_ARGS PKG_CONFIG_NAME arrow-metal)
endif()
add_arrow_lib(arrow_metal
              CMAKE_PACKAGE_NAME
              ArrowMetal
              ${ARROW_METAL_PKG_CONFIG_NAME_ARGS}
              SOURCES
              ${ARROW_METAL_SRCS}
              OUTPUTS
              ARROW_METAL_LIBRARIES
              SHARED_LINK_FLAGS
              ${ARROW_VERSION_SCRIPT_FLAGS}  # defined in cpp/arrow/CMakeLists.txt
              SHARED_LINK_LIBS
              ${ARROW_METAL_LINK_LIBS}
              arrow_shared
              ${ARROW_METAL_SHARED_LINK_LIBS}
              SHARED_INSTALL_INTERFACE_LIBS
              Arrow::arrow_shared
              ${ARROW_METAL_SHARED_LINK_LIBS}
              # Static arrow_metal must also link against the system frameworks
              STATIC_LINK_LIBS
              ${ARROW_METAL_LINK_LIBS}
              ${ARROW_METAL_SHARED_LINK_LIBS}
              STATIC_INSTALL_INTERFACE_LIBS
              Arrow::arrow_static
              ${ARROW_METAL_SHARED_LINK_LIBS})

add_dependencies(arrow_metal ${ARROW_METAL_LIBRARIES})

foreach(LIB_TARGET ${ARROW_METAL_LIBRARIES})
  target_compile_definitions(${LIB_TARGET} PRIVATE ARROW_METAL_EXPORTING)
endforeach()

# Metal SDK build version header.
configure_file(metal_version.h.in
               "${CMAKE_CURRENT_BINARY_DIR}/metal_version.h" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/metal_version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/arrow/metal")

arrow_install_all_headers("arrow/metal")

if(ARROW_BUILD_SHARED)
  set(ARROW_METAL_LIBRARY arrow_metal_shared)
else()
  set(ARROW_METAL_LIBRARY arrow_metal_static)
endif()

set(ARROW_METAL_TEST_LINK_LIBS ${ARROW_METAL_LIBRARY} ${ARROW_TEST_LINK_LIBS})

if(ARROW_BUILD_TESTS)
  # metal_test exists as a .mm because some cases dispatch a real Metal
  # compute kernel inline to validate GPU coherence. add_arrow_test takes
  # an explicit SOURCES override so the .cc default doesn't apply.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/metal_test.mm")
    set_source_files_properties(metal_test.mm
                                PROPERTIES COMPILE_FLAGS "-fobjc-arc"
                                           SKIP_PRECOMPILE_HEADERS ON)
    add_arrow_test(metal_test
                   SOURCES
                   metal_test.mm
                   STATIC_LINK_LIBS
                   ${ARROW_METAL_TEST_LINK_LIBS}
                   NO_VALGRIND)
  endif()
endif()

if(ARROW_BUILD_BENCHMARKS)
  add_arrow_benchmark(metal_benchmark
                      PREFIX
                      "arrow-metal"
                      EXTRA_LINK_LIBS
                      ${ARROW_METAL_LIBRARY})
  add_dependencies(arrow_metal-benchmarks arrow-metal-metal-benchmark)
endif()
