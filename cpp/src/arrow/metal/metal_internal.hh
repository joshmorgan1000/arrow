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

// arrow_metal internal bridge header.
//
// This file is .hh on purpose: it must NEVER be included from a public
// header, only from .mm translation units that already pull Metal/Foundation
// in. Public headers under arrow/metal/ store Apple objects as `void*` so
// downstream C++ consumers can include them from plain .cpp files.
//
// Bridges between an `id<...>` and `void*`:
//
//   void* opaque = ToOpaque(mtl_buffer);          // retained
//   id<MTLBuffer> back = FromOpaque<MTLBuffer>(o); // borrowed
//   ReleaseOpaque(opaque);                         // matched Release
//
// The retain count discipline matches std::shared_ptr semantics: ownership
// hops produce one balanced retain/release.

#pragma once

#ifndef __OBJC__
#  error "metal_internal.hh must only be included from Objective-C++ (.mm)"
#endif
#ifndef __APPLE__
#  error "metal_internal.hh is Apple-only"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>

namespace arrow {
namespace metal {
namespace internal {

/// \brief Transfer an ARC-managed Objective-C object to an opaque void*
///
/// Increments the retain count by one. The returned pointer must later
/// be balanced with a call to `ReleaseOpaque`, otherwise the object
/// leaks. Use this when an ObjC object needs to outlive the current
/// scope and be reattached on the .mm side via `FromOpaqueRetained`.
template <typename ObjCType>
inline void* ToOpaqueRetained(ObjCType obj) {
  return (__bridge_retained void*)obj;
}

/// \brief Reattach an opaque void* to an ARC-managed handle (consuming)
///
/// The opaque pointer must originate from `ToOpaqueRetained`. Ownership
/// transfers to the returned handle; the caller MUST NOT call
/// `ReleaseOpaque` on the same pointer afterwards.
template <typename ObjCType>
inline ObjCType FromOpaqueConsuming(void* opaque) {
  return (__bridge_transfer ObjCType)opaque;
}

/// \brief Borrow a non-owning handle from an opaque void*
///
/// Does not change the retain count. The returned handle is valid only
/// while `opaque` keeps the object alive (i.e. while a balancing
/// `ReleaseOpaque` has not yet run).
template <typename ObjCType>
inline ObjCType FromOpaqueBorrowed(void* opaque) {
  return (__bridge ObjCType)opaque;
}

/// \brief Release an opaque pointer obtained from `ToOpaqueRetained`
///
/// Balances the retain done by `ToOpaqueRetained`. Safe to call with
/// nullptr.
inline void ReleaseOpaque(void* opaque) {
  if (opaque == nullptr) {
    return;
  }
  // __bridge_transfer hands ownership to ARC; the implicit dtor at
  // scope exit releases.
  (void)(__bridge_transfer id)opaque;
}

}  // namespace internal
}  // namespace metal
}  // namespace arrow
