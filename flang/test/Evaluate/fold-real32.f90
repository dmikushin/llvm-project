! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!
! REAL(KIND=32) is IEEE binary256: p = 237, w = 19, bias = 262143.
!
! The values below are not transcribed from anywhere. They follow from the
! interchange-format rule of IEEE 754-2019 clause 3.6 for k = 256, and the
! same rule reproduces binary128's 113/15/16383 - which is how it was
! checked before being written down here.
!
! The two divisions are the load-bearing cases. A quotient that does not
! terminate in binary has last bits only if the arithmetic really carried 237
! of them, so folding this expression at kind 16 and widening the result
! produces a value that differs from the one below at bit 114. An exactly
! representable constant would agree at every width and would test nothing.
!
! The digits here were produced by MPFR at precision 237 with the binary256
! exponent range, independently of flang; audit/fold32 in the liboctamath
! repository regenerates and re-checks them, and demonstrates that its own
! judge fails when one digit is perturbed.

subroutine properties
  integer, parameter :: oct = selected_real_kind(70)
! CHECK: INTEGER, PARAMETER :: oct = 32_4
  integer, parameter :: beyond = selected_real_kind(72)
! CHECK: INTEGER, PARAMETER :: beyond = -1_4
  integer, parameter :: p = precision(1.0_oct)
! CHECK: INTEGER, PARAMETER :: p = 71_4
  integer, parameter :: r = range(1.0_oct)
! CHECK: INTEGER, PARAMETER :: r = 78912_4
  integer, parameter :: d = digits(1.0_oct)
! CHECK: INTEGER, PARAMETER :: d = 237_4
  integer, parameter :: mn = minexponent(1.0_oct)
! CHECK: INTEGER, PARAMETER :: mn = -262141_4
  integer, parameter :: mx = maxexponent(1.0_oct)
! CHECK: INTEGER, PARAMETER :: mx = 262144_4
end subroutine

subroutine folding
  integer, parameter :: oct = selected_real_kind(70)
  real(oct), parameter :: third = 1.0_oct / 3.0_oct
! CHECK: PARAMETER :: third = 3.33333333333333333333333333333333333333333333333333333333333333333333332578693410097
  real(oct), parameter :: eps = epsilon(1.0_oct)
! CHECK: PARAMETER :: eps = 9.05567907882671236750911929088779178068253119813913818958261488993550131859284511473
  real(oct), parameter :: onepluseps = 1.0_oct + eps
! CHECK: PARAMETER :: onepluseps = 1.00000000000000000000000000000000000000000000000000000000000000000000000905567907882
end subroutine
