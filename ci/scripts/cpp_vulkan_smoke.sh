#!/usr/bin/env bash
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

# Local smoke check for ARROW_VULKAN: configure, build only the vulkan
# target + its tests, run them. Works on macOS (via MoltenVK), Linux
# with Mesa drivers, or anywhere libvulkan.so.1 + libvulkan-dev are
# available.
#
# Usage:
#   ci/scripts/cpp_vulkan_smoke.sh [build_dir]
#
# If unset, build_dir defaults to /tmp/arrow-vulkan-check.
#
# Override `ARROW_VULKAN_DEPS` to control dependency resolution:
#   AUTO    (default) — system / Homebrew where available, BUNDLED otherwise
#   BUNDLED            — build everything from source (slow first run)
#   SYSTEM             — fail if system dep is missing

set -euo pipefail

BUILD_DIR="${1:-/tmp/arrow-vulkan-check}"
SOURCE_DIR="$(cd "$(dirname "$0")/../.." && pwd)/cpp"
ARROW_VULKAN_DEPS="${ARROW_VULKAN_DEPS:-AUTO}"

cmake -GNinja \
  -S "${SOURCE_DIR}" \
  -B "${BUILD_DIR}" \
  -DARROW_VULKAN=ON \
  -DARROW_BUILD_TESTS=ON \
  -DARROW_BUILD_BENCHMARKS=OFF \
  -DARROW_BUILD_SHARED=ON \
  -DARROW_BUILD_STATIC=OFF \
  -DARROW_DEPENDENCY_SOURCE="${ARROW_VULKAN_DEPS}" \
  -DARROW_IPC=ON \
  -DARROW_JEMALLOC=OFF \
  -DARROW_MIMALLOC=OFF \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "${BUILD_DIR}" arrow-vulkan-test
ctest --test-dir "${BUILD_DIR}" -R arrow-vulkan --output-on-failure
