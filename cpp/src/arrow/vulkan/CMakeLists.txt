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
# arrow_vulkan
#
# Host-coherent Vulkan buffer / device / memory-manager backend for
# Apache Arrow. Targets integrated GPUs (AMD APU, Intel iGPU, Mali,
# Adreno, MoltenVK on Apple Silicon, llvmpipe software fallback) where
# `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` memory exists. Allocations
# from arrow_vulkan are simultaneously CPU- and GPU-addressable with no
# staging copy. Discrete GPUs without ReBAR are out of scope and rejected
# at device selection time with a clear error.

add_custom_target(arrow_vulkan-all)
add_custom_target(arrow_vulkan)
add_custom_target(arrow_vulkan-benchmarks)
add_custom_target(arrow_vulkan-tests)
add_dependencies(arrow_vulkan-all arrow_vulkan arrow_vulkan-tests
                 arrow_vulkan-benchmarks)

# `find_package(Vulkan 1.2 REQUIRED)` lives at the top-level cmake under
# `if(ARROW_VULKAN)` so the Vulkan::Vulkan target is already available
# when this directory is descended into.

# pkg-config private cflags. Static-link consumers need ARROW_VULKAN_STATIC.
if(NOT ARROW_BUILD_SHARED AND ARROW_BUILD_STATIC)
  string(APPEND ARROW_VULKAN_PC_CFLAGS "${ARROW_VULKAN_PC_CFLAGS_PRIVATE}")
  set(ARROW_VULKAN_PC_CFLAGS_PRIVATE "")
endif()

# Sources. Each task in the vulkan-native plan appends to this list as
# it lands its files.
set(ARROW_VULKAN_SRCS vulkan_init.cc
                      vulkan_buffer.cc
                      vulkan_device.cc
                      vulkan_memory.cc
                      vulkan_memory_pool.cc)

set(ARROW_VULKAN_LINK_LIBS arrow::flatbuffers)
set(ARROW_VULKAN_SHARED_LINK_LIBS Vulkan::Vulkan)

set(ARROW_VULKAN_PKG_CONFIG_NAME_ARGS)
if(NOT WINDOWS)
  list(APPEND ARROW_VULKAN_PKG_CONFIG_NAME_ARGS PKG_CONFIG_NAME arrow-vulkan)
endif()
add_arrow_lib(arrow_vulkan
              CMAKE_PACKAGE_NAME
              ArrowVulkan
              ${ARROW_VULKAN_PKG_CONFIG_NAME_ARGS}
              SOURCES
              ${ARROW_VULKAN_SRCS}
              OUTPUTS
              ARROW_VULKAN_LIBRARIES
              SHARED_LINK_FLAGS
              ${ARROW_VERSION_SCRIPT_FLAGS}  # defined in cpp/arrow/CMakeLists.txt
              SHARED_LINK_LIBS
              ${ARROW_VULKAN_LINK_LIBS}
              arrow_shared
              ${ARROW_VULKAN_SHARED_LINK_LIBS}
              SHARED_INSTALL_INTERFACE_LIBS
              Arrow::arrow_shared
              ${ARROW_VULKAN_SHARED_LINK_LIBS}
              # Static arrow_vulkan must also link against Vulkan
              STATIC_LINK_LIBS
              ${ARROW_VULKAN_LINK_LIBS}
              ${ARROW_VULKAN_SHARED_LINK_LIBS}
              STATIC_INSTALL_INTERFACE_LIBS
              Arrow::arrow_static
              ${ARROW_VULKAN_SHARED_LINK_LIBS})

add_dependencies(arrow_vulkan ${ARROW_VULKAN_LIBRARIES})

foreach(LIB_TARGET ${ARROW_VULKAN_LIBRARIES})
  target_compile_definitions(${LIB_TARGET} PRIVATE ARROW_VULKAN_EXPORTING)
endforeach()

# Vulkan loader version header.
configure_file(vulkan_version.h.in
               "${CMAKE_CURRENT_BINARY_DIR}/vulkan_version.h" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vulkan_version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/arrow/vulkan")

arrow_install_all_headers("arrow/vulkan")

if(ARROW_BUILD_SHARED)
  set(ARROW_VULKAN_LIBRARY arrow_vulkan_shared)
else()
  set(ARROW_VULKAN_LIBRARY arrow_vulkan_static)
endif()

set(ARROW_VULKAN_TEST_LINK_LIBS ${ARROW_VULKAN_LIBRARY} ${ARROW_TEST_LINK_LIBS})

if(ARROW_BUILD_TESTS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_test.cc")
    add_arrow_test(vulkan_test STATIC_LINK_LIBS ${ARROW_VULKAN_TEST_LINK_LIBS}
                               NO_VALGRIND)
  endif()
endif()

if(ARROW_BUILD_BENCHMARKS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_benchmark.cc")
    add_arrow_benchmark(vulkan_benchmark
                        PREFIX
                        "arrow-vulkan"
                        EXTRA_LINK_LIBS
                        ${ARROW_VULKAN_LIBRARY})
    add_dependencies(arrow_vulkan-benchmarks arrow-vulkan-vulkan-benchmark)
  endif()
endif()
