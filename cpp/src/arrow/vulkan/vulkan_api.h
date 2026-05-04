// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

/// \file
/// \brief Aggregate header for arrow_vulkan — pulls in every public type.
///
/// Plain C++ consumers can `#include "arrow/vulkan/vulkan_api.h"` from
/// any .cc file (no <vulkan/vulkan.h> required).

#include "arrow/vulkan/vulkan_buffer.h"        // IWYU pragma: export
#include "arrow/vulkan/vulkan_device.h"        // IWYU pragma: export
#include "arrow/vulkan/vulkan_memory.h"        // IWYU pragma: export
#include "arrow/vulkan/vulkan_memory_pool.h"   // IWYU pragma: export
