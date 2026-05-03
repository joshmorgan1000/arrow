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

# Local smoke check for ARROW_METAL: configure, build only the metal
# target + its tests, run them, and clean up. Intended for fast iteration
# on macOS while developing arrow_metal.
#
# Usage:
#   ci/scripts/cpp_metal_smoke.sh [build_dir]
#
# If unset, build_dir defaults to /tmp/arrow-metal-check.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: cpp_metal_smoke.sh requires macOS (ARROW_METAL is Apple-only)" >&2
  exit 1
fi

BUILD_DIR="${1:-/tmp/arrow-metal-check}"
SOURCE_DIR="$(cd "$(dirname "$0")/../.." && pwd)/cpp"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -GNinja \
  -DARROW_METAL=ON \
  -DARROW_BUILD_TESTS=ON \
  -DARROW_BUILD_BENCHMARKS=OFF \
  -DARROW_BUILD_SHARED=ON \
  -DARROW_BUILD_STATIC=OFF \
  -DARROW_DEPENDENCY_SOURCE=BUNDLED \
  -DARROW_IPC=ON \
  -DARROW_JEMALLOC=OFF \
  -DARROW_MIMALLOC=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  "${SOURCE_DIR}"

ninja arrow-metal-test
"./release/arrow-metal-test"
