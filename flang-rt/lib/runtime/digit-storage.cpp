//===-- lib/runtime/digit-storage.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The runtime's half of the digit-storage seam declared in
// flang/lib/Decimal/big-radix-floating-point.h.
//
// The runtime is freestanding and has no operator new[], which is what stopped
// the decimal conversion from being instantiated here at binary256 and so
// stopped the numeric edit descriptors from working on REAL(32). It does have
// an allocator of its own, and that is what this uses. The compiler's
// definitions live in flang/lib/Decimal/digit-storage.cpp, which is not
// compiled into flang-rt.

#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Common/api-attrs.h"
#include <cstddef>

namespace Fortran::decimal {

RT_API_ATTRS void *DecimalDigitStorageAllocate(std::size_t bytes) {
  runtime::Terminator terminator{__FILE__, __LINE__};
  return runtime::AllocateMemoryOrCrash(terminator, bytes);
}

RT_API_ATTRS void DecimalDigitStorageFree(void *p) { runtime::FreeMemory(p); }

} // namespace Fortran::decimal
