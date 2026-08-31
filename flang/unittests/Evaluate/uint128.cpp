#define AVOID_NATIVE_UINT128_T 1
#include "flang/Common/uint128.h"
#include "flang/Testing/testing.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>

#if (defined __GNUC__ || defined __clang__) && defined __SIZEOF_INT128__
#define HAS_NATIVE_UINT128_T 1
#else
#undef HAS_NATIVE_UINT128_T
#endif

using U128 = Fortran::common::UnsignedInt128;

static void Test(std::uint64_t x) {
  U128 n{x};
  MATCH(x, static_cast<std::uint64_t>(n));
  MATCH(~x, static_cast<std::uint64_t>(~n));
  MATCH(-x, static_cast<std::uint64_t>(-n));
  MATCH(!x, static_cast<std::uint64_t>(!n));
  TEST(n == n);
  TEST(n + n == n * static_cast<U128>(2));
  TEST(n - n == static_cast<U128>(0));
  TEST(n + n == n << static_cast<U128>(1));
  TEST(n + n == n << static_cast<U128>(1));
  TEST((n + n) - n == n);
  TEST(((n + n) >> static_cast<U128>(1)) == n);
  if (x != 0) {
    TEST(static_cast<U128>(0) / n == static_cast<U128>(0));
    TEST(static_cast<U128>(n - 1) / n == static_cast<U128>(0));
    TEST(static_cast<U128>(n) / n == static_cast<U128>(1));
    TEST(static_cast<U128>(n + n - 1) / n == static_cast<U128>(1));
    TEST(static_cast<U128>(n + n) / n == static_cast<U128>(2));
  }
}

static void Test(std::uint64_t x, std::uint64_t y) {
  U128 m{x}, n{y};
  MATCH(x, static_cast<std::uint64_t>(m));
  MATCH(y, static_cast<std::uint64_t>(n));
  MATCH(x & y, static_cast<std::uint64_t>(m & n));
  MATCH(x | y, static_cast<std::uint64_t>(m | n));
  MATCH(x ^ y, static_cast<std::uint64_t>(m ^ n));
  MATCH(x + y, static_cast<std::uint64_t>(m + n));
  MATCH(x - y, static_cast<std::uint64_t>(m - n));
  MATCH(x * y, static_cast<std::uint64_t>(m * n));
  if (n != 0) {
    MATCH(x / y, static_cast<std::uint64_t>(m / n));
  }
}

// The 256-bit type is the same template with 128-bit halves.  Because this
// file defines AVOID_NATIVE_UINT128_T, those halves are themselves the
// portable UnsignedInt128, so these cases exercise the fully composed path -
// two levels of hand-written halves - which is where an off-by-one at the
// seam between halves would hide.
using U256 = Fortran::common::UnsignedInt256;

static U256 Make256(std::uint64_t w3, std::uint64_t w2, std::uint64_t w1,
    std::uint64_t w0) {
  U256 v{w3};
  v <<= 64;
  v |= U256{w2};
  v <<= 64;
  v |= U256{w1};
  v <<= 64;
  v |= U256{w0};
  return v;
}

static void Test256() {
  TEST(sizeof(U256) * 8 >= 256);
  TEST(U256::bits == 256);

  // A single bit at every position a shift can land on or straddle.  63/64/65
  // and 191/192 cross the halves of the halves; 127/128/129 cross the halves.
  for (int s{0}; s < 256; ++s) {
    U256 bit{U256{1u} << s};
    TEST(bit != U256{0u});
    TEST((bit >> s) == U256{1u});
    // Exactly one bit set: subtracting one clears it and sets all below.
    TEST((bit & (bit - U256{1u})) == U256{0u});
    // Shifting off the top gives zero rather than wrapping around.
    TEST((bit << (256 - s)) == U256{0u});
  }
  TEST((U256{1u} << 256) == U256{0u});

  // Words that all differ, so a swapped or duplicated half is visible.
  U256 a{Make256(0x0123456789abcdefull, 0xfedcba9876543210ull,
      0x00000000ffffffffull, 0xffffffff00000000ull)};
  U256 b{Make256(0x1111111111111111ull, 0x2222222222222222ull,
      0x3333333333333333ull, 0x4444444444444444ull)};

  // The halves must reassemble into the whole.
  TEST(((a >> 128) << 128 | (a & ((U256{1u} << 128) - U256{1u}))) == a);

  TEST((a ^ a) == U256{0u});
  TEST((a & ~a) == U256{0u});
  TEST((a | ~a) == ~U256{0u});
  TEST((a + b) - b == a);
  TEST((a - b) + b == a);
  TEST(-a + a == U256{0u});
  TEST(~a + U256{1u} == -a);

  // Multiplication against shifting, which reaches the quarter-split path.
  for (int s{0}; s < 200; ++s) {
    TEST(b * (U256{1u} << s) == (b << s));
  }

  // Division and remainder against their definition.
  TEST(a / b * b + a % b == a);
  TEST(a / b == U256{0u}); // a < b: their leading words are 0x0123... and 0x1111...
  TEST(b / a == Make256(0u, 0u, 0u, 0xfull));
  TEST(a % U256{1u} == U256{0u});
  TEST(a / U256{1u} == a);
  TEST(a / a == U256{1u});
  TEST(a % a == U256{0u});

  TEST(a < b);
  TEST(b > a);
  TEST(a >= a);
  TEST(a <= a);
  TEST(a != b);

  // Computed with Python at arbitrary precision, so that the reference for
  // these is not the implementation being tested.
  //   a = 0x0123456789abcdeffedcba98765432100000_0000ffffffffffffffff00000000
  //   b = 0x1111111111111111222222222222222233333333333333334444444444444444
  TEST(a + b ==
      Make256(0x123456789abcdf01ull, 0x20fedcba98765432ull,
          0x3333333433333333ull, 0x4444444344444444ull));
  TEST(a - b ==
      Make256(0xf0123456789abcdeull, 0xdcba987654320fedull,
          0xcccccccdccccccccull, 0xbbbbbbbabbbbbbbcull));
  TEST(a * b ==
      Make256(0x21c10aff9ee8dd7cull, 0xcd1a78e868fa9d51ull,
          0x11111110bbbbbbbbull, 0xbbbbbbbc00000000ull));
}

#if HAS_NATIVE_UINT128_T
static __uint128_t ToNative(U128 n) {
  return static_cast<__uint128_t>(static_cast<std::uint64_t>(n >> 64)) << 64 |
      static_cast<std::uint64_t>(n);
}

static U128 FromNative(__uint128_t n) {
  return U128{static_cast<std::uint64_t>(n >> 64)} << 64 |
      U128{static_cast<std::uint64_t>(n)};
}

static void TestVsNative(__uint128_t x, __uint128_t y) {
  U128 m{FromNative(x)}, n{FromNative(y)};
  TEST(ToNative(m) == x);
  TEST(ToNative(n) == y);
  TEST(ToNative(~m) == ~x);
  TEST(ToNative(-m) == -x);
  TEST(ToNative(!m) == !x);
  TEST(ToNative(m < n) == (x < y));
  TEST(ToNative(m <= n) == (x <= y));
  TEST(ToNative(m == n) == (x == y));
  TEST(ToNative(m != n) == (x != y));
  TEST(ToNative(m >= n) == (x >= y));
  TEST(ToNative(m > n) == (x > y));
  TEST(ToNative(m & n) == (x & y));
  TEST(ToNative(m | n) == (x | y));
  TEST(ToNative(m ^ n) == (x ^ y));
  if (y < 128) {
    TEST(ToNative(m << n) == (x << y));
    TEST(ToNative(m >> n) == (x >> y));
  }
  TEST(ToNative(m + n) == (x + y));
  TEST(ToNative(m - n) == (x - y));
  TEST(ToNative(m * n) == (x * y));
  if (y > 0) {
    TEST(ToNative(m / n) == (x / y));
    TEST(ToNative(m % n) == (x % y));
    TEST(ToNative(m - n * (m / n)) == (x % y));
  }
}

static void TestVsNative() {
  for (int j{0}; j < 128; ++j) {
    for (int k{0}; k < 128; ++k) {
      __uint128_t m{1}, n{1};
      m <<= j, n <<= k;
      TestVsNative(m, n);
      TestVsNative(~m, n);
      TestVsNative(m, ~n);
      TestVsNative(~m, ~n);
      TestVsNative(m ^ n, n);
      TestVsNative(m, m ^ n);
      TestVsNative(m ^ ~n, n);
      TestVsNative(m, ~m ^ n);
      TestVsNative(m ^ ~n, m ^ n);
      TestVsNative(m ^ n, ~m ^ n);
      TestVsNative(m ^ ~n, ~m ^ n);
      Test(m, 10000000000000000); // important case for decimal conversion
      Test(~m, 10000000000000000);
    }
  }
}
#endif

int main() {
  for (std::uint64_t j{0}; j < 64; ++j) {
    Test(j);
    Test(~j);
    Test(std::uint64_t(1) << j);
    for (std::uint64_t k{0}; k < 64; ++k) {
      Test(j, k);
    }
  }
  Test256();
#if HAS_NATIVE_UINT128_T
  llvm::outs() << "Environment has native __uint128_t\n";
  TestVsNative();
#else
  llvm::outs() << "Environment lacks native __uint128_t\n";
#endif
  return testing::Complete();
}
