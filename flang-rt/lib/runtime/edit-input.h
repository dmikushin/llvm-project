//===-- lib/runtime/edit-input.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_EDIT_INPUT_H_
#define FLANG_RT_RUNTIME_EDIT_INPUT_H_

#include "flang-rt/runtime/format.h"
#include "flang-rt/runtime/io-stmt.h"
#include "flang/Decimal/decimal.h"

namespace Fortran::runtime::io {

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS bool EditIntegerInput(
    IoStatementState &, const DataEdit &, void *, int kind, bool isSigned);

template <int KIND>
RT_API_ATTRS bool EditRealInput(IoStatementState &, const DataEdit &, void *);

/// Scan one REAL input field and hand back its decimal text.
///
/// This exists for REAL(32). Every other kind is converted inside the runtime
/// by EditRealInput, which ends in decimal::ConvertToBinary at a precision
/// instantiated there; binary256 is not among them, because the buffers that
/// width needs would have to come from a heap the freestanding runtime does
/// not have. So the division of labour is the same one list-directed output
/// already uses: the runtime owns records, fields and the input format, and
/// the caller owns the decimal-to-binary conversion.
///
/// The text written is what EditRealInput would have converted - a normalised
/// fraction with an optional leading '-', a radix point, and an 'e' exponent
/// when one is needed - and it is NUL-terminated. That form is ordinary
/// decimal notation, so any correctly rounding parser reproduces the value
/// the runtime would have produced.
///
/// Returns false on a malformed field, on a hexadecimal one (0X..., which is
/// an extension this path does not carry), and when the field needs more
/// characters than the buffer holds. The last case signals an error rather
/// than truncating: a truncated decimal string is a different number, and
/// silently reading a different number is the failure this whole line of work
/// exists to prevent.
RT_API_ATTRS bool ScanRealInputToDecimal(
    IoStatementState &, const DataEdit &, char *buffer, std::size_t bufferSize);

RT_API_ATTRS bool EditLogicalInput(
    IoStatementState &, const DataEdit &, bool &);

template <typename CHAR>
RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, CHAR *, std::size_t);

extern template RT_API_ATTRS bool EditRealInput<2>(
    IoStatementState &, const DataEdit &, void *);
extern template RT_API_ATTRS bool EditRealInput<3>(
    IoStatementState &, const DataEdit &, void *);
extern template RT_API_ATTRS bool EditRealInput<4>(
    IoStatementState &, const DataEdit &, void *);
extern template RT_API_ATTRS bool EditRealInput<8>(
    IoStatementState &, const DataEdit &, void *);
extern template RT_API_ATTRS bool EditRealInput<10>(
    IoStatementState &, const DataEdit &, void *);
// TODO: double/double
extern template RT_API_ATTRS bool EditRealInput<16>(
    IoStatementState &, const DataEdit &, void *);
#if !defined(RT_DEVICE_COMPILATION)
extern template RT_API_ATTRS bool EditRealInput<32>(
    IoStatementState &, const DataEdit &, void *);
#endif

extern template RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char *, std::size_t);
extern template RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char16_t *, std::size_t);
extern template RT_API_ATTRS bool EditCharacterInput(
    IoStatementState &, const DataEdit &, char32_t *, std::size_t);

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime::io
#endif // FLANG_RT_RUNTIME_EDIT_INPUT_H_
