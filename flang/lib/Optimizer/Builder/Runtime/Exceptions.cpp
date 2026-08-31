//===-- Exceptions.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Exceptions.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/exceptions.h"

using namespace Fortran::runtime;

mlir::Value fir::runtime::genMapExcept(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(MapException)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func, excepts).getResult(0);
}

void fir::runtime::genFeclearexcept(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(feclearexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

void fir::runtime::genFeraiseexcept(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(feraiseexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

mlir::Value fir::runtime::genFetestexcept(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(fetestexcept)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func, excepts).getResult(0);
}

void fir::runtime::genFedisableexcept(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(fedisableexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

void fir::runtime::genFeenableexcept(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(feenableexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

mlir::Value fir::runtime::genFegetexcept(fir::FirOpBuilder &builder,
                                         mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(fegetexcept)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

mlir::Value fir::runtime::genSupportHalting(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(SupportHalting)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func, excepts).getResult(0);
}

mlir::Value fir::runtime::genGetUnderflowMode(fir::FirOpBuilder &builder,
                                              mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(GetUnderflowMode)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

void fir::runtime::genSetUnderflowMode(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value flag) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(SetUnderflowMode)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, flag);
}

mlir::Value fir::runtime::genGetModesTypeSize(fir::FirOpBuilder &builder,
                                              mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(GetModesTypeSize)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

mlir::Value fir::runtime::genGetStatusTypeSize(fir::FirOpBuilder &builder,
                                               mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(GetStatusTypeSize)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

void fir::runtime::genRaiseOctaStatus(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value status) {
  mlir::Type i32 = builder.getI32Type();

  // liboctamath returns its own bit set; flang's ieee_flag_type encoding has a
  // DENORM bit at position 1 that liboctamath does not have, so everything
  // above INVALID sits one place higher there:
  //
  //   octa  invalid=1 divzero=2 overflow=4 underflow=8  inexact=16
  //   flang invalid=1 denorm=2  divzero=4  overflow=8   underflow=16 inexact=32
  //
  // Bit 0 stays where it is and bits 1..4 shift up by one. Bits above that are
  // dropped on purpose: OCTA_MEMORY is not an IEEE exception, and
  // OCTA_ZIV_EXHAUSTED - "faithful, but possibly not correctly rounded" - has
  // no IEEE flag to land on. Mapping it onto INEXACT would be true and would
  // also make it indistinguishable from ordinary rounding, which is exactly
  // the distinction the library goes to some trouble to preserve.
  mlir::Value invalidMask = builder.createIntegerConstant(loc, i32, 0x1);
  mlir::Value shiftedMask = builder.createIntegerConstant(loc, i32, 0x1e);
  mlir::Value one = builder.createIntegerConstant(loc, i32, 1);
  mlir::Value low =
      mlir::arith::AndIOp::create(builder, loc, status, invalidMask);
  mlir::Value high = mlir::arith::ShLIOp::create(
      builder, loc,
      mlir::arith::AndIOp::create(builder, loc, status, shiftedMask), one);
  mlir::Value flags = mlir::arith::OrIOp::create(builder, loc, low, high);

  // Guarded so that code raising nothing costs nothing. This is not an
  // optimisation of the common case - most real arithmetic is inexact and will
  // take the branch - it keeps an exact operation from paying for two runtime
  // calls that would have no effect.
  mlir::Value zero = builder.createIntegerConstant(loc, i32, 0);
  mlir::Value any = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, flags, zero);
  builder.genIfThen(loc, any)
      .genThen([&]() {
        genFeraiseexcept(builder, loc, genMapExcept(builder, loc, flags));
      })
      .end();
}
