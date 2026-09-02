//===-- Exceptions.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H

#include "mlir/IR/Value.h"

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate a runtime call to map a set of ieee_flag_type exceptions to a
/// libm fenv.h excepts value.
mlir::Value genMapExcept(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value excepts);

void genFeclearexcept(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value excepts);

void genFeraiseexcept(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value excepts);

mlir::Value genFetestexcept(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value excepts);

void genFedisableexcept(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value excepts);

void genFeenableexcept(fir::FirOpBuilder &builder, mlir::Location loc,
                       mlir::Value excepts);

mlir::Value genFegetexcept(fir::FirOpBuilder &builder, mlir::Location loc);

mlir::Value genSupportHalting(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value excepts);

mlir::Value genGetUnderflowMode(fir::FirOpBuilder &builder, mlir::Location loc);
void genSetUnderflowMode(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value bit);

mlir::Value genGetModesTypeSize(fir::FirOpBuilder &builder, mlir::Location loc);
mlir::Value genGetStatusTypeSize(fir::FirOpBuilder &builder,
                                 mlir::Location loc);

/// Raise the IEEE exceptions named by a liboctamath status word.
///
/// REAL(32) arithmetic happens in a software library, so its exceptions never
/// reach the hardware status word that IEEE_GET_FLAG reads through
/// fetestexcept. This translates the library's own bit set into flang's
/// ieee_flag_type encoding and raises it, which is what makes the
/// IEEE_ARITHMETIC module observe binary256 arithmetic at all rather than
/// report a permanently clear set.
void genRaiseOctaStatus(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value status);

/// Translate a rounding mode from flang's encoding into liboctamath's.
///
/// flang uses llvm.get.rounding's numbering, which is also what
/// magic-numbers.h gives IEEE_ROUND_TYPE: 0 to-zero, 1 nearest, 2 up, 3 down,
/// 4 away. liboctamath numbers them 0 nearest-even, 1 zero, 2 down, 3 up,
/// 4 nearest-away. The first four disagree pairwise, so this is a translation
/// and emphatically not a cast: passing one encoding as the other silently
/// swaps to-zero with nearest and up with down.
mlir::Value genOctaRoundingMode(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value mode);

/// The rounding mode in force right now, in liboctamath's encoding.
///
/// Hardware kinds obey the FPU control word, so their arithmetic passes
/// nothing. REAL(32) is computed by a library that takes the mode per call, so
/// every call site has to read it. Emitting a constant instead makes
/// IEEE_SET_ROUNDING_MODE silently ineffective for this kind alone - measured:
/// under IEEE_UP and then IEEE_DOWN, 1/3 compared equal at REAL(32) while
/// REAL(8) correctly differed.
mlir::Value genOctaCurrentRoundingMode(fir::FirOpBuilder &builder,
                                       mlir::Location loc);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
