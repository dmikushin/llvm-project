//===-- lib/Decimal/digit-storage.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The compiler's half of the digit-storage seam declared in
// big-radix-floating-point.h. flang-rt compiles binary-to-decimal.cpp and
// decimal-to-binary.cpp out of this directory but deliberately not this file;
// it supplies its own definitions in flang-rt/lib/runtime/digit-storage.cpp,
// which allocate through the runtime's allocator instead. One definition per
// binary, so the template above has a single code path.

#include "big-radix-floating-point.h"
#include <cstdlib>

namespace Fortran::decimal {

void *DecimalDigitStorageAllocate(std::size_t bytes) {
  void *p{std::malloc(bytes)};
  if (!p) {
    // LLVM builds without exceptions, so this cannot report std::bad_alloc.
    // The operator new[] this replaced would have called std::terminate for
    // the same reason, so aborting is the behaviour that was already there.
    std::abort();
  }
  return p;
}

void DecimalDigitStorageFree(void *p) { std::free(p); }

} // namespace Fortran::decimal
