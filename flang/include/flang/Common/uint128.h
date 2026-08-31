//===-- include/flang/Common/uint128.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Portable 128-bit and 256-bit integer arithmetic.
//
// The 128-bit type exists for C++ implementations lacking __uint128_t and
// __int128_t.  The 256-bit type exists because no implementation has a native
// one: it is needed to hold an IEEE-754 binary256 value in
// decimal::BinaryFloatingPointNumber, which keeps the whole encoding in a
// single unsigned integer.
//
// Both come from one implementation, parameterized on the type of each half.
// Writing a separate 256-bit class would have duplicated the shift, carry and
// division logic - the places where an off-by-one across the halves hides -
// and left two copies to keep in agreement.

#ifndef FORTRAN_COMMON_UINT128_H_
#define FORTRAN_COMMON_UINT128_H_

// Define AVOID_NATIVE_UINT128_T to force the use of UnsignedInt128 below
// instead of the C++ compiler's native 128-bit unsigned integer type, if
// it has one.
#ifndef AVOID_NATIVE_UINT128_T
#define AVOID_NATIVE_UINT128_T 0
#endif

#include "api-attrs.h"
#include "leading-zero-bit-count.h"
#include <climits>
#include <cstdint>
#include <type_traits>

namespace Fortran::common {

// Counting leading zeroes in a native 128-bit integer, needed by the division
// of a type whose halves are that wide.  It must be declared before the class
// template that calls it: the argument is a builtin type, so argument-dependent
// lookup finds nothing at the point of instantiation, and only declarations
// visible where the template is defined are considered.  Declared after, the
// call is ambiguous between the narrower overloads instead.
#if (defined __GNUC__ || defined __clang__) && defined __SIZEOF_INT128__
inline constexpr int LeadingZeroBitCount(__uint128_t x) {
  std::uint64_t hi{static_cast<std::uint64_t>(x >> 64)};
  if (hi == 0) {
    return 64 + LeadingZeroBitCount(static_cast<std::uint64_t>(x));
  } else {
    return LeadingZeroBitCount(hi);
  }
}
#endif

// An integer of twice the width of HALF, held as two halves.  HALF must be an
// unsigned integer type - either a native one or a narrower instantiation of
// this template, which is how the 256-bit type is built on the 128-bit one.
template <typename HALF, bool IS_SIGNED = false> class IntBase {
public:
  static constexpr int halfBits{CHAR_BIT * sizeof(HALF)};
  static constexpr int bits{2 * halfBits};

  constexpr IntBase() {}
  // This means of definition provides some portability for
  // "size_t" operands.
  constexpr IntBase(unsigned n) : low_{n} {}
  constexpr IntBase(unsigned long n) : low_{n} {}
  constexpr IntBase(unsigned long long n) : low_{n} {}
  constexpr IntBase(int n) {
    low_ = static_cast<HALF>(n);
    high_ = n < 0 ? ~HALF{0} : HALF{0};
  }
  constexpr IntBase(long n) {
    low_ = static_cast<HALF>(n);
    high_ = n < 0 ? ~HALF{0} : HALF{0};
  }
  constexpr IntBase(long long n) {
    low_ = static_cast<HALF>(n);
    high_ = n < 0 ? ~HALF{0} : HALF{0};
  }
  // Construction from the half type itself.  Only declared when HALF is wider
  // than the widest builtin the constructors above accept, i.e. only for the
  // 256-bit instantiation; declaring it unconditionally would make
  // IntBase<std::uint64_t>{someUint64} ambiguous.
  template <typename H = HALF,
      typename = std::enable_if_t<(sizeof(H) > sizeof(unsigned long long))>>
  constexpr IntBase(H n) : low_{n} {}

  constexpr IntBase(const IntBase &) = default;
  constexpr IntBase(IntBase &&) = default;
  constexpr IntBase &operator=(const IntBase &) = default;
  constexpr IntBase &operator=(IntBase &&) = default;

  explicit constexpr IntBase(const IntBase<HALF, !IS_SIGNED> &n)
      : low_{n.low()}, high_{n.high()} {}
  explicit constexpr IntBase(IntBase<HALF, !IS_SIGNED> &&n)
      : low_{n.low()}, high_{n.high()} {}

  constexpr IntBase operator+() const { return *this; }
  constexpr IntBase operator~() const { return {~high_, ~low_}; }
  constexpr IntBase operator-() const { return ~*this + 1; }
  constexpr bool operator!() const { return !low_ && !high_; }
  constexpr explicit operator bool() const {
    return low_ != HALF{0} || high_ != HALF{0};
  }
  constexpr explicit operator std::uint64_t() const {
    return static_cast<std::uint64_t>(low_);
  }
  constexpr explicit operator std::int64_t() const {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(low_));
  }
  constexpr explicit operator int() const {
    return static_cast<int>(static_cast<std::uint64_t>(low_));
  }

  constexpr HALF high() const { return high_; }
  constexpr HALF low() const { return low_; }

  constexpr IntBase operator++(/*prefix*/) {
    *this += 1;
    return *this;
  }
  constexpr IntBase operator++(int /*postfix*/) {
    IntBase result{*this};
    *this += 1;
    return result;
  }
  constexpr IntBase operator--(/*prefix*/) {
    *this -= 1;
    return *this;
  }
  constexpr IntBase operator--(int /*postfix*/) {
    IntBase result{*this};
    *this -= 1;
    return result;
  }

  constexpr IntBase operator&(IntBase that) const {
    return {high_ & that.high_, low_ & that.low_};
  }
  constexpr IntBase operator|(IntBase that) const {
    return {high_ | that.high_, low_ | that.low_};
  }
  constexpr IntBase operator^(IntBase that) const {
    return {high_ ^ that.high_, low_ ^ that.low_};
  }

  constexpr IntBase operator<<(IntBase that) const {
    if (that >= bits) {
      return {};
    } else if (that == 0) {
      return *this;
    } else {
      int n{static_cast<int>(that.low_)};
      if (n >= halfBits) {
        return {low_ << (n - halfBits), HALF{0}};
      } else {
        return {(high_ << n) | (low_ >> (halfBits - n)), low_ << n};
      }
    }
  }
  constexpr IntBase operator>>(IntBase that) const {
    if (that >= bits) {
      return {};
    } else if (that == 0) {
      return *this;
    } else {
      int n{static_cast<int>(that.low_)};
      if (n >= halfBits) {
        return {HALF{0}, high_ >> (n - halfBits)};
      } else {
        return {high_ >> n, (high_ << (halfBits - n)) | (low_ >> n)};
      }
    }
  }

  constexpr IntBase operator+(IntBase that) const {
    HALF lower{(low_ & ~topBit) + (that.low_ & ~topBit)};
    bool carry{((lower >> (halfBits - 1)) + (low_ >> (halfBits - 1)) +
                   (that.low_ >> (halfBits - 1))) > HALF{1}};
    return {static_cast<HALF>(high_ + that.high_ + HALF{carry}),
        static_cast<HALF>(low_ + that.low_)};
  }
  constexpr IntBase operator-(IntBase that) const { return *this + -that; }

  constexpr IntBase operator*(IntBase that) const {
    // Split each half into two quarters so that the partial products of two
    // quarters fit in one half without overflow.
    constexpr int quarterBits{halfBits / 2};
    const HALF maskQuarter{static_cast<HALF>((HALF{1} << quarterBits) - 1)};
    if (high_ == HALF{0} && that.high_ == HALF{0}) {
      HALF x0{low_ & maskQuarter}, x1{low_ >> quarterBits};
      HALF y0{that.low_ & maskQuarter}, y1{that.low_ >> quarterBits};
      IntBase x0y0{static_cast<HALF>(x0 * y0)},
          x0y1{static_cast<HALF>(x0 * y1)};
      IntBase x1y0{static_cast<HALF>(x1 * y0)},
          x1y1{static_cast<HALF>(x1 * y1)};
      return x0y0 + ((x0y1 + x1y0) << quarterBits) + (x1y1 << halfBits);
    } else {
      HALF x0{low_ & maskQuarter}, x1{low_ >> quarterBits},
          x2{high_ & maskQuarter}, x3{high_ >> quarterBits};
      HALF y0{that.low_ & maskQuarter}, y1{that.low_ >> quarterBits},
          y2{that.high_ & maskQuarter}, y3{that.high_ >> quarterBits};
      IntBase x0y0{static_cast<HALF>(x0 * y0)},
          x0y1{static_cast<HALF>(x0 * y1)},
          x0y2{static_cast<HALF>(x0 * y2)},
          x0y3{static_cast<HALF>(x0 * y3)};
      IntBase x1y0{static_cast<HALF>(x1 * y0)},
          x1y1{static_cast<HALF>(x1 * y1)},
          x1y2{static_cast<HALF>(x1 * y2)};
      IntBase x2y0{static_cast<HALF>(x2 * y0)},
          x2y1{static_cast<HALF>(x2 * y1)};
      IntBase x3y0{static_cast<HALF>(x3 * y0)};
      return x0y0 + ((x0y1 + x1y0) << quarterBits) +
          ((x0y2 + x1y1 + x2y0) << halfBits) +
          ((x0y3 + x1y2 + x2y1 + x3y0) << (halfBits + quarterBits));
    }
  }

  constexpr IntBase operator/(IntBase that) const {
    int j{LeadingZeroes()};
    IntBase bitsLeft{*this};
    bitsLeft <<= j;
    IntBase numerator{};
    IntBase quotient{};
    for (; j < bits; ++j) {
      numerator <<= 1;
      if (bitsLeft.high_ & topBit) {
        numerator.low_ |= HALF{1};
      }
      bitsLeft <<= 1;
      quotient <<= 1;
      if (numerator >= that) {
        ++quotient;
        numerator -= that;
      }
    }
    return quotient;
  }

  constexpr IntBase operator%(IntBase that) const {
    int j{LeadingZeroes()};
    IntBase bitsLeft{*this};
    bitsLeft <<= j;
    IntBase remainder{};
    for (; j < bits; ++j) {
      remainder <<= 1;
      if (bitsLeft.high_ & topBit) {
        remainder.low_ |= HALF{1};
      }
      bitsLeft <<= 1;
      if (remainder >= that) {
        remainder -= that;
      }
    }
    return remainder;
  }

  constexpr bool operator<(IntBase that) const {
    if (IS_SIGNED && ((high_ ^ that.high_) & topBit) != HALF{0}) {
      return (high_ & topBit) != HALF{0};
    }
    return high_ < that.high_ || (high_ == that.high_ && low_ < that.low_);
  }
  constexpr bool operator<=(IntBase that) const { return !(*this > that); }
  constexpr bool operator==(IntBase that) const {
    return low_ == that.low_ && high_ == that.high_;
  }
  constexpr bool operator!=(IntBase that) const { return !(*this == that); }
  constexpr bool operator>=(IntBase that) const { return that <= *this; }
  constexpr bool operator>(IntBase that) const { return that < *this; }

  constexpr IntBase &operator&=(const IntBase &that) {
    *this = *this & that;
    return *this;
  }
  constexpr IntBase &operator|=(const IntBase &that) {
    *this = *this | that;
    return *this;
  }
  constexpr IntBase &operator^=(const IntBase &that) {
    *this = *this ^ that;
    return *this;
  }
  constexpr IntBase &operator<<=(const IntBase &that) {
    *this = *this << that;
    return *this;
  }
  constexpr IntBase &operator>>=(const IntBase &that) {
    *this = *this >> that;
    return *this;
  }
  constexpr IntBase &operator+=(const IntBase &that) {
    *this = *this + that;
    return *this;
  }
  constexpr IntBase &operator-=(const IntBase &that) {
    *this = *this - that;
    return *this;
  }
  constexpr IntBase &operator*=(const IntBase &that) {
    *this = *this * that;
    return *this;
  }
  constexpr IntBase &operator/=(const IntBase &that) {
    *this = *this / that;
    return *this;
  }
  constexpr IntBase &operator%=(const IntBase &that) {
    *this = *this % that;
    return *this;
  }

private:
  constexpr IntBase(HALF hi, HALF lo) {
    low_ = lo;
    high_ = hi;
  }
  constexpr int LeadingZeroes() const {
    if (high_ == HALF{0}) {
      return halfBits + LeadingZeroBitCount(low_);
    } else {
      return LeadingZeroBitCount(high_);
    }
  }
  RT_VAR_GROUP_BEGIN
  static constexpr HALF topBit{HALF{1} << (halfBits - 1)};
  RT_VAR_GROUP_END
#if FLANG_LITTLE_ENDIAN
  HALF low_{0}, high_{0};
#elif FLANG_BIG_ENDIAN
  HALF high_{0}, low_{0};
#else
#error host endianness is not known
#endif
};

// The 128-bit type keeps its historical name and interface; it is now one
// instantiation of the template above.
template <bool IS_SIGNED = false>
using Int128 = IntBase<std::uint64_t, IS_SIGNED>;

using UnsignedInt128 = Int128<false>;
using SignedInt128 = Int128<true>;

// Counting leading zeroes in a half is needed by division.  Providing it for
// the composed types keeps the 256-bit instantiation complete rather than
// leaving division to fail at the first use.
template <typename HALF, bool IS_SIGNED>
inline constexpr int LeadingZeroBitCount(const IntBase<HALF, IS_SIGNED> &x) {
  if (x.high() == HALF{0}) {
    return (CHAR_BIT * sizeof(HALF)) + LeadingZeroBitCount(x.low());
  } else {
    return LeadingZeroBitCount(x.high());
  }
}

#if !AVOID_NATIVE_UINT128_T && (defined __GNUC__ || defined __clang__) && \
    defined __SIZEOF_INT128__
#define USING_NATIVE_INT128_T 1
using uint128_t = __uint128_t;
using int128_t = __int128_t;
#else
#undef USING_NATIVE_INT128_T
using uint128_t = UnsignedInt128;
using int128_t = SignedInt128;
#endif

// IEEE-754 binary256 needs all 256 bits of its encoding in one unsigned
// integer; no implementation has a native one, so it is always composed.
using UnsignedInt256 = IntBase<uint128_t, false>;
using SignedInt256 = IntBase<uint128_t, true>;
using uint256_t = UnsignedInt256;
using int256_t = SignedInt256;

template <int BITS> struct HostUnsignedIntTypeHelper {
  using type = std::conditional_t<(BITS <= 8), std::uint8_t,
      std::conditional_t<(BITS <= 16), std::uint16_t,
          std::conditional_t<(BITS <= 32), std::uint32_t,
              std::conditional_t<(BITS <= 64), std::uint64_t,
                  std::conditional_t<(BITS <= 128), uint128_t, uint256_t>>>>>;
};
template <int BITS> struct HostSignedIntTypeHelper {
  using type = std::conditional_t<(BITS <= 8), std::int8_t,
      std::conditional_t<(BITS <= 16), std::int16_t,
          std::conditional_t<(BITS <= 32), std::int32_t,
              std::conditional_t<(BITS <= 64), std::int64_t,
                  std::conditional_t<(BITS <= 128), int128_t, int256_t>>>>>;
};
template <int BITS>
using HostUnsignedIntType = typename HostUnsignedIntTypeHelper<BITS>::type;
template <int BITS>
using HostSignedIntType = typename HostSignedIntTypeHelper<BITS>::type;

} // namespace Fortran::common
#endif
