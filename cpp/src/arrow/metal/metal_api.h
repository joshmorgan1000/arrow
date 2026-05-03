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
/// \brief Aggregate header for arrow_metal — pulls in every public type.
///
/// Plain C++ consumers can `#include "arrow/metal/metal_api.h"` from
/// any .cpp file (no Objective-C++ required).

#include "arrow/metal/metal_buffer.h"        // IWYU pragma: export
#include "arrow/metal/metal_device.h"        // IWYU pragma: export
#include "arrow/metal/metal_memory.h"        // IWYU pragma: export
#include "arrow/metal/metal_memory_pool.h"   // IWYU pragma: export
