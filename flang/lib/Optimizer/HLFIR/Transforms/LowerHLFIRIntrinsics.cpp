//===- LowerHLFIRIntrinsics.cpp - Transformational intrinsics to FIR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_LOWERHLFIRINTRINSICS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

/// Base class for passes converting transformational intrinsic operations into
/// runtime calls
template <class OP>
class HlfirIntrinsicConversion : public mlir::OpRewritePattern<OP> {
public:
  explicit HlfirIntrinsicConversion(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<OP>{ctx} {
    // required for cases where intrinsics are chained together e.g.
    // matmul(matmul(a, b), c)
    // because converting the inner operation then invalidates the
    // outer operation: causing the pattern to apply recursively.
    //
    // This is safe because we always progress with each iteration. Circular
    // applications of operations are not expressible in MLIR because we use
    // an SSA form and one must become first. E.g.
    // %a = hlfir.matmul %b %d
    // %b = hlfir.matmul %a %d
    // cannot be written.
    // MSVC needs the this->
    this->setHasBoundedRewriteRecursion(true);
  }

protected:
  struct IntrinsicArgument {
    mlir::Value val; // allowed to be null if the argument is absent
    mlir::Type desiredType;
  };

  /// Lower the arguments to the intrinsic: adding necessary boxing and
  /// conversion to match the signature of the intrinsic in the runtime library.
  llvm::SmallVector<fir::ExtendedValue, 3>
  lowerArguments(mlir::Operation *op,
                 const llvm::ArrayRef<IntrinsicArgument> &args,
                 mlir::PatternRewriter &rewriter,
                 const fir::IntrinsicArgumentLoweringRules *argLowering) const {
    mlir::Location loc = op->getLoc();
    fir::FirOpBuilder builder{rewriter, op};

    llvm::SmallVector<fir::ExtendedValue, 3> ret;
    llvm::SmallVector<std::function<void()>, 2> cleanupFns;

    for (size_t i = 0; i < args.size(); ++i) {
      mlir::Value arg = args[i].val;
      mlir::Type desiredType = args[i].desiredType;
      if (!arg) {
        ret.emplace_back(fir::getAbsentIntrinsicArgument());
        continue;
      }
      hlfir::Entity entity{arg};

      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, i);
      switch (argRules.lowerAs) {
      case fir::LowerIntrinsicArgAs::Value: {
        if (args[i].desiredType != arg.getType()) {
          arg = builder.createConvert(loc, desiredType, arg);
          entity = hlfir::Entity{arg};
        }
        auto [exv, cleanup] = hlfir::convertToValue(loc, builder, entity);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(exv);
      } break;
      case fir::LowerIntrinsicArgAs::Addr: {
        auto [exv, cleanup] =
            hlfir::convertToAddress(loc, builder, entity, desiredType);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(exv);
      } break;
      case fir::LowerIntrinsicArgAs::Box: {
        auto [box, cleanup] =
            hlfir::convertToBox(loc, builder, entity, desiredType);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(box);
      } break;
      case fir::LowerIntrinsicArgAs::Inquired: {
        if (args[i].desiredType != arg.getType()) {
          arg = builder.createConvert(loc, desiredType, arg);
          entity = hlfir::Entity{arg};
        }
        // Place hlfir.expr in memory, and unbox fir.boxchar. Other entities
        // are translated to fir::ExtendedValue without transofrmation (notably,
        // pointers/allocatable are not dereferenced).
        // TODO: once lowering to FIR retires, UBOUND and LBOUND can be
        // simplified since the fir.box lowered here are now guarenteed to
        // contain the local lower bounds thanks to the hlfir.declare (the extra
        // rebox can be removed).
        // When taking arguments as descriptors, the runtime expect absent
        // OPTIONAL to be a nullptr to a descriptor, lowering has already
        // prepared such descriptors as needed, hence set
        // keepScalarOptionalBoxed to avoid building descriptors with a null
        // address for them.
        auto [exv, cleanup] = hlfir::translateToExtendedValue(
            loc, builder, entity, /*contiguous=*/false,
            /*keepScalarOptionalBoxed=*/true);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(exv);
      } break;
      }
    }

    if (cleanupFns.size()) {
      auto oldInsertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointAfter(op);
      for (std::function<void()> cleanup : cleanupFns)
        cleanup();
      builder.restoreInsertionPoint(oldInsertionPoint);
    }

    return ret;
  }

  void processReturnValue(mlir::Operation *op,
                          const fir::ExtendedValue &resultExv, bool mustBeFreed,
                          fir::FirOpBuilder &builder,
                          mlir::PatternRewriter &rewriter) const {
    mlir::Location loc = op->getLoc();

    mlir::Value firBase = fir::getBase(resultExv);
    mlir::Type firBaseTy = firBase.getType();

    std::optional<hlfir::EntityWithAttributes> resultEntity;
    if (fir::isa_trivial(firBaseTy)) {
      // Some intrinsics return i1 when the original operation
      // produces fir.logical<>, so we may need to cast it.
      firBase = builder.createConvert(loc, op->getResult(0).getType(), firBase);
      resultEntity = hlfir::EntityWithAttributes{firBase};
    } else {
      resultEntity =
          hlfir::genDeclare(loc, builder, resultExv, ".tmp.intrinsic_result",
                            fir::FortranVariableFlagsAttr{});
    }

    if (resultEntity->isVariable()) {
      hlfir::AsExprOp asExpr = hlfir::AsExprOp::create(
          builder, loc, *resultEntity, builder.createBool(loc, mustBeFreed));
      resultEntity = hlfir::EntityWithAttributes{asExpr.getResult()};
    }

    mlir::Value base = resultEntity->getBase();
    if (!mlir::isa<hlfir::ExprType>(base.getType())) {
      for (mlir::Operation *use : op->getResult(0).getUsers()) {
        if (mlir::isa<hlfir::DestroyOp>(use))
          rewriter.eraseOp(use);
      }
    }

    rewriter.replaceOp(op, base);
  }
};

// Given an integer or array of integer type, calculate the Kind parameter from
// the width for use in runtime intrinsic calls.
static unsigned getKindForType(mlir::Type ty) {
  mlir::Type eltty = hlfir::getFortranElementType(ty);
  unsigned width = mlir::cast<mlir::IntegerType>(eltty).getWidth();
  return width / 8;
}

// --- REAL(32) reductions -------------------------------------------------
//
// REAL(32) is IEEE binary256 carried as an opaque i256, so there is no runtime
// kernel to call: the runtime's reductions are templated over a C++ element
// type and binary256 has none. Every operation on the type is a call into
// liboctamath, and a reduction is no different - it is a loop of them.
//
// This is emitted here, in the always-run lowering, rather than in
// SimplifyHLFIRIntrinsics where the loop machinery already exists, because
// that pass only runs above -O0. A reduction that is inlined at -O2 and
// dispatched to a runtime kernel at -O0 would give this type two different
// implementations, and the defect this replaced was exactly a difference
// between optimisation levels: SUM of four 1.0_oct returned -2.014e78912 at
// -O2 and refused to compile at -O0.
//
// Scope: total reductions without a MASK. DIM and MASK still take the runtime
// path and still fail loudly there, which is the honest outcome for something
// unimplemented - see the arr32 test for what is covered.

/// Emit `int octa_<name>(octa_t *r, const octa_t *a, const octa_t *b, int)`
/// and return the loaded result. Mirrors genLibOctaCall in IntrinsicCall.cpp:
/// the library takes and returns binary256 by pointer, so the operands are
/// spilled to allocas.
static mlir::Value genOctaBinary(mlir::Location loc, fir::FirOpBuilder &builder,
                                 llvm::StringRef name, mlir::Value a,
                                 mlir::Value b) {
  mlir::Type i256 = builder.getIntegerType(256);
  mlir::Type ref = fir::ReferenceType::get(i256);
  mlir::Type i32 = builder.getI32Type();
  llvm::SmallVector<mlir::Type> argTys{ref, ref, ref, i32};
  mlir::func::FuncOp fn = builder.getNamedFunction(name);
  if (!fn)
    fn = builder.createFunction(loc, name,
                                builder.getFunctionType(argTys, {i32}));

  mlir::Value result = fir::AllocaOp::create(builder, loc, i256);
  mlir::Value slotA = fir::AllocaOp::create(builder, loc, i256);
  mlir::Value slotB = fir::AllocaOp::create(builder, loc, i256);
  fir::StoreOp::create(builder, loc, a, slotA);
  fir::StoreOp::create(builder, loc, b, slotB);
  mlir::Value rnd = builder.createIntegerConstant(loc, i32, 0);
  fir::CallOp::create(builder, loc, fn,
                      mlir::ValueRange{result, slotA, slotB, rnd});
  return fir::LoadOp::create(builder, loc, result);
}

/// Emit `int octa_cmp(const octa_t *a, const octa_t *b, int *unordered)` and
/// return its -1/0/1 code. If \p unorderedOut is non-null it receives the
/// unordered flag, which is reported separately rather than folded into the
/// code precisely so that a caller cannot ignore NaN by accident.
static mlir::Value genOctaCmp(mlir::Location loc, fir::FirOpBuilder &builder,
                              mlir::Value a, mlir::Value b,
                              mlir::Value *unorderedOut = nullptr) {
  mlir::Type i256 = builder.getIntegerType(256);
  mlir::Type ref = fir::ReferenceType::get(i256);
  mlir::Type i32 = builder.getI32Type();
  mlir::Type i32ref = fir::ReferenceType::get(i32);
  mlir::func::FuncOp fn = builder.getNamedFunction("octa_cmp");
  if (!fn)
    fn = builder.createFunction(
        loc, "octa_cmp", builder.getFunctionType({ref, ref, i32ref}, {i32}));

  mlir::Value slotA = fir::AllocaOp::create(builder, loc, i256);
  mlir::Value slotB = fir::AllocaOp::create(builder, loc, i256);
  mlir::Value unordered = fir::AllocaOp::create(builder, loc, i32);
  fir::StoreOp::create(builder, loc, a, slotA);
  fir::StoreOp::create(builder, loc, b, slotB);
  mlir::Value code = fir::CallOp::create(builder, loc, fn,
                                         mlir::ValueRange{slotA, slotB,
                                                          unordered})
                         .getResult(0);
  if (unorderedOut)
    *unorderedOut = fir::LoadOp::create(builder, loc, unordered);
  return code;
}

/// A binary256 constant, given as the four 64-bit words of the interchange
/// encoding with w3 first. Spelled from the format rather than borrowed from
/// a header so that the bit positions are visible where they are used:
/// sign in bit 63 of w3, then 19 exponent bits, bias 262143.
static mlir::Value genOcta(mlir::Location loc, fir::FirOpBuilder &builder,
                           uint64_t w3, uint64_t rest) {
  llvm::APInt v(256, w3);
  v <<= 192;
  if (rest)
    v |= llvm::APInt::getLowBitsSet(256, 192);
  return mlir::arith::ConstantOp::create(
      builder, loc, builder.getIntegerType(256),
      builder.getIntegerAttr(builder.getIntegerType(256), v));
}

template <class OP>
class HlfirReductionIntrinsicConversion : public HlfirIntrinsicConversion<OP> {
  using HlfirIntrinsicConversion<OP>::HlfirIntrinsicConversion;
  using IntrinsicArgument =
      typename HlfirIntrinsicConversion<OP>::IntrinsicArgument;
  using HlfirIntrinsicConversion<OP>::lowerArguments;
  using HlfirIntrinsicConversion<OP>::processReturnValue;

protected:
  auto buildNumericalArgs(OP operation, mlir::Type i32, mlir::Type logicalType,
                          mlir::PatternRewriter &rewriter,
                          std::string opName) const {
    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({operation.getArray(), operation.getArray().getType()});
    inArgs.push_back({operation.getDim(), i32});
    inArgs.push_back({operation.getMask(), logicalType});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  auto buildMinMaxLocArgs(OP operation, mlir::Type i32, mlir::Type logicalType,
                          mlir::PatternRewriter &rewriter, std::string opName,
                          fir::FirOpBuilder builder) const {
    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({operation.getArray(), operation.getArray().getType()});
    inArgs.push_back({operation.getDim(), i32});
    inArgs.push_back({operation.getMask(), logicalType});
    mlir::Value kind = builder.createIntegerConstant(
        operation->getLoc(), i32, getKindForType(operation.getType()));
    inArgs.push_back({kind, i32});
    inArgs.push_back({operation.getBack(), i32});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  auto buildLogicalArgs(OP operation, mlir::Type i32, mlir::Type logicalType,
                        mlir::PatternRewriter &rewriter,
                        std::string opName) const {
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({operation.getMask(), logicalType});
    inArgs.push_back({operation.getDim(), i32});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  /// The identity of the reduction, as a binary256 bit pattern.
  ///
  /// 1.0 has exponent 262143 = 0x3FFFF in bits 62..44 and a zero fraction; an
  /// infinity has the exponent all ones, 0x7FFFF; HUGE has 0x7FFFE and every
  /// fraction bit set. The sign is bit 63. MAXVAL and MINVAL seed with an
  /// infinity rather than -/+ HUGE, because seeding with HUGE would return
  /// HUGE for an array whose elements are all infinite; the standard's empty
  /// answer is put back afterwards, so neither case is traded for the other.
  mlir::Value octaInit(mlir::Location loc, fir::FirOpBuilder &builder) const {
    if constexpr (std::is_same_v<OP, hlfir::SumOp>)
      return genOcta(loc, builder, 0, 0);
    else if constexpr (std::is_same_v<OP, hlfir::ProductOp>)
      return genOcta(loc, builder, 0x3FFFF00000000000ULL, 0);
    else if constexpr (std::is_same_v<OP, hlfir::MaxvalOp>)
      return genOcta(loc, builder, 0xFFFFF00000000000ULL, 0); // -infinity
    else
      return genOcta(loc, builder, 0x7FFFF00000000000ULL, 0); // +infinity
  }

  /// One accumulation step, shared by the total reduction and the DIM form.
  ///
  /// SUM carries a compensation term as a second value. That is not a
  /// refinement of my own: flang-rt/lib/runtime/sum.cpp accumulates every
  /// other real kind with Kahan summation, and a REAL(32) that summed naively
  /// would be the one kind that behaves differently. The extra width is not an
  /// argument against it - binary64 has 53 bits and compensates, and the
  /// cancellation Kahan repairs grows with the number of terms, not with the
  /// format.
  llvm::SmallVector<mlir::Value>
  octaStep(mlir::Location l, fir::FirOpBuilder &b, hlfir::Entity array,
           mlir::ValueRange idx, mlir::ValueRange acc) const {
    constexpr bool isSum = std::is_same_v<OP, hlfir::SumOp>;
    constexpr bool isProduct = std::is_same_v<OP, hlfir::ProductOp>;
    constexpr bool isMax = std::is_same_v<OP, hlfir::MaxvalOp>;

    hlfir::Entity elem = hlfir::loadElementAt(l, b, array, idx);
    mlir::Value cur = acc[0];
    if constexpr (isSum) {
      mlir::Value corr = acc[1];
      mlir::Value next = genOctaBinary(l, b, "octa_sub", elem, corr);
      // Both arms are computed and selected rather than branched on: each is a
      // handful of library calls, and control flow here would have to be
      // threaded through the reduction values by hand.
      mlir::Value sumIfNan = genOctaBinary(l, b, "octa_add", cur, elem);
      mlir::Value sumOk = genOctaBinary(l, b, "octa_add", cur, next);
      mlir::Value diff = genOctaBinary(l, b, "octa_sub", sumOk, cur);
      mlir::Value corrOk = genOctaBinary(l, b, "octa_sub", diff, next);
      // next != next, i.e. an Inf - Inf that produced a NaN correction.
      mlir::Value unordered;
      (void)genOctaCmp(l, b, next, next, &unordered);
      mlir::Value zeroI32 = b.createIntegerConstant(l, b.getI32Type(), 0);
      mlir::Value isNan = mlir::arith::CmpIOp::create(
          b, l, mlir::arith::CmpIPredicate::ne, unordered, zeroI32);
      mlir::Value zero = genOcta(l, b, 0, 0);
      return {mlir::arith::SelectOp::create(b, l, isNan, sumIfNan, sumOk),
              mlir::arith::SelectOp::create(b, l, isNan, zero, corrOk)};
    } else if constexpr (isProduct) {
      return {genOctaBinary(l, b, "octa_mul", cur, elem)};
    } else {
      mlir::Value code = genOctaCmp(l, b, elem, cur);
      mlir::Value zero = b.createIntegerConstant(l, b.getI32Type(), 0);
      mlir::Value take = mlir::arith::CmpIOp::create(
          b, l,
          isMax ? mlir::arith::CmpIPredicate::sgt
                : mlir::arith::CmpIPredicate::slt,
          code, zero);
      return {mlir::arith::SelectOp::create(b, l, take, elem, cur)};
    }
  }

  /// Total SUM/PRODUCT/MAXVAL/MINVAL over a REAL(32) array, as a loop of
  /// liboctamath calls.
  ///
  /// The empty-array answers are the standard's and are not what the loop
  /// produces: SUM is 0, PRODUCT is 1, MAXVAL is -HUGE and MINVAL is +HUGE.
  /// For MAXVAL and MINVAL the loop is seeded with -/+ infinity instead,
  /// because seeding with -/+ HUGE would return HUGE for an array whose
  /// elements are all infinite. The two are reconciled after the loop by
  /// selecting on whether the array had any elements at all, so neither case
  /// is traded for the other.
  llvm::LogicalResult genOctaReduction(OP operation, llvm::StringRef opName,
                                       fir::FirOpBuilder &builder,
                                       mlir::Location loc, hlfir::Entity array,
                                       mlir::Value mask,
                                       mlir::PatternRewriter &rewriter) const {
    constexpr bool isSum = std::is_same_v<OP, hlfir::SumOp>;
    constexpr bool isProduct = std::is_same_v<OP, hlfir::ProductOp>;
    constexpr bool isMax = std::is_same_v<OP, hlfir::MaxvalOp>;

    mlir::Value init = octaInit(loc, builder);

    llvm::SmallVector<mlir::Value> extents =
        hlfir::genExtentsVector(loc, builder, array);

    llvm::SmallVector<mlir::Value> inits{init};
    if constexpr (isSum)
      inits.push_back(genOcta(loc, builder, 0, 0)); // correction

    // A scalar mask does not vary with the element, so it is read once. The
    // Entity is built only when there is a mask, because hlfir::Entity
    // dereferences in its constructor and a null one crashes.
    std::optional<hlfir::Entity> maskEntity;
    mlir::Value scalarMask;
    if (mask) {
      maskEntity = hlfir::Entity{mask};
      if (maskEntity->isScalar())
        scalarMask = builder.createConvert(
            loc, builder.getI1Type(),
            hlfir::loadTrivialScalar(loc, builder, *maskEntity));
    }

    // With a mask, MAXVAL and MINVAL need to know whether anything was
    // selected at all, which is not the same question as whether the array
    // was empty: an array of a thousand elements with every one masked out
    // must still give -HUGE. The flag is carried only when it is needed, so
    // the unmasked loop is unchanged.
    const bool needTaken = mask && !isSum && !isProduct;
    if (needTaken)
      inits.push_back(builder.createIntegerConstant(loc, builder.getI1Type(), 0));

    auto step = [&](mlir::Location l, fir::FirOpBuilder &b,
                    mlir::ValueRange idx,
                    mlir::ValueRange acc) -> llvm::SmallVector<mlir::Value> {
      return octaStep(l, b, array, idx, acc);
    };

    // The mask is applied by selecting between the stepped accumulator and the
    // one that came in, rather than by branching: every arm here is a handful
    // of library calls, and control flow would have to be threaded through the
    // reduction values by hand. A masked-out element therefore leaves every
    // carried value exactly as it was, which is what "not present in the
    // reduction" means for SUM's compensation term as much as for its sum.
    auto body = [&](mlir::Location l, fir::FirOpBuilder &b,
                    mlir::ValueRange idx,
                    mlir::ValueRange acc) -> llvm::SmallVector<mlir::Value> {
      if (!mask)
        return step(l, b, idx, acc);

      mlir::Value m = scalarMask;
      if (!m) {
        hlfir::Entity mElem = hlfir::loadElementAt(l, b, *maskEntity, idx);
        m = b.createConvert(l, b.getI1Type(), mElem);
      }

      const unsigned carried = needTaken ? acc.size() - 1 : acc.size();
      llvm::SmallVector<mlir::Value> in(acc.begin(), acc.begin() + carried);
      llvm::SmallVector<mlir::Value> next = step(l, b, idx, in);
      llvm::SmallVector<mlir::Value> out;
      for (unsigned k = 0; k < carried; ++k)
        out.push_back(mlir::arith::SelectOp::create(b, l, m, next[k], in[k]));
      if (needTaken)
        out.push_back(
            mlir::arith::OrIOp::create(b, l, acc[carried], m));
      return out;
    };

    llvm::SmallVector<mlir::Value> result = hlfir::genLoopNestWithReductions(
        loc, builder, extents, inits, body, /*isUnordered=*/false);
    mlir::Value reduced = result[0];

    if constexpr (!isSum && !isProduct) {
      // Replace the infinity seed with the standard's empty-array answer when
      // the array turned out to be empty. Without this MAXVAL of a
      // zero-sized array returns -infinity where Fortran says -HUGE.
      mlir::Value count = builder.createIntegerConstant(
          loc, builder.getIndexType(), 1);
      for (mlir::Value e : extents)
        count = mlir::arith::MulIOp::create(builder, loc, count, e);
      mlir::Value zeroIdx =
          builder.createIntegerConstant(loc, builder.getIndexType(), 0);
      mlir::Value empty = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, count, zeroIdx);
      // With a mask the question is not whether the array was empty but
      // whether anything was selected; an array whose every element is masked
      // out must give the same answer as an empty one.
      if (needTaken) {
        mlir::Value one =
            builder.createIntegerConstant(loc, builder.getI1Type(), 1);
        empty = mlir::arith::XOrIOp::create(builder, loc, result.back(), one);
      }
      mlir::Value huge =
          isMax ? genOcta(loc, builder, 0xFFFFEFFFFFFFFFFFULL, 1)
                : genOcta(loc, builder, 0x7FFFEFFFFFFFFFFFULL, 1);
      reduced =
          mlir::arith::SelectOp::create(builder, loc, empty, huge, reduced);
    }

    rewriter.replaceOp(operation, reduced);
    return mlir::success();
  }

  /// SUM/PRODUCT/MAXVAL/MINVAL over one dimension of a REAL(32) array.
  ///
  /// The result is rank n-1: one reduction per output element, each over the
  /// single contracted extent. The accumulation step is the one the total
  /// reduction uses, called through octaStep rather than copied, because two
  /// copies of a compensated sum are two things that can disagree about a
  /// digit - and the disagreement would appear only with DIM, which is the
  /// half nobody looks at.
  ///
  /// DIM must be a compile-time constant. A runtime DIM would mean choosing
  /// which axis to contract while the loop nest is being built, and that is a
  /// different construction rather than a parameter of this one; it is refused
  /// before code is generated instead of being approximated here.
  llvm::LogicalResult genOctaReductionDim(OP operation, fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          hlfir::Entity array, mlir::Value mask,
                                          unsigned dim,
                                          mlir::PatternRewriter &rewriter) const {
    constexpr bool isSum = std::is_same_v<OP, hlfir::SumOp>;
    constexpr bool isProduct = std::is_same_v<OP, hlfir::ProductOp>;
    constexpr bool isMax = std::is_same_v<OP, hlfir::MaxvalOp>;

    llvm::SmallVector<mlir::Value> extents =
        hlfir::genExtentsVector(loc, builder, array);
    const unsigned rank = extents.size();
    const unsigned axis = dim - 1;

    mlir::Type idxTy = builder.getIndexType();
    mlir::Type eleTy = builder.getIntegerType(256);

    // The result shape is the argument's with the contracted extent removed.
    llvm::SmallVector<mlir::Value> resExtents;
    for (unsigned k = 0; k < rank; ++k)
      if (k != axis)
        resExtents.push_back(extents[k]);
    mlir::Value shape = builder.genShape(loc, resExtents);

    std::optional<hlfir::Entity> maskEntity;
    mlir::Value scalarMask;
    if (mask) {
      maskEntity = hlfir::Entity{mask};
      if (maskEntity->isScalar())
        scalarMask = builder.createConvert(
            loc, builder.getI1Type(),
            hlfir::loadTrivialScalar(loc, builder, *maskEntity));
    }
    const bool needTaken = mask && !isSum && !isProduct;

    hlfir::ElementalOp elemental = hlfir::genElementalOp(
        loc, builder, eleTy, shape, /*typeParams=*/{},
        [&](mlir::Location l, fir::FirOpBuilder &b,
            mlir::ValueRange outIdx) -> hlfir::Entity {
          llvm::SmallVector<mlir::Value> inits{octaInit(l, b)};
          if constexpr (isSum)
            inits.push_back(genOcta(l, b, 0, 0));
          if (needTaken)
            inits.push_back(b.createIntegerConstant(l, b.getI1Type(), 0));

          auto body = [&](mlir::Location bl, fir::FirOpBuilder &bb,
                          mlir::ValueRange inner,
                          mlir::ValueRange acc)
              -> llvm::SmallVector<mlir::Value> {
            // Rebuild the argument's full subscript by putting the contracted
            // loop's index back at the axis it came from.
            llvm::SmallVector<mlir::Value> full;
            for (unsigned k = 0, o = 0; k < rank; ++k)
              full.push_back(k == axis ? inner[0] : outIdx[o++]);

            const unsigned carried = needTaken ? acc.size() - 1 : acc.size();
            llvm::SmallVector<mlir::Value> in(acc.begin(),
                                              acc.begin() + carried);
            llvm::SmallVector<mlir::Value> next =
                octaStep(bl, bb, array, full, in);
            if (!mask)
              return next;

            mlir::Value m = scalarMask;
            if (!m) {
              hlfir::Entity mElem =
                  hlfir::loadElementAt(bl, bb, *maskEntity, full);
              m = bb.createConvert(bl, bb.getI1Type(), mElem);
            }
            llvm::SmallVector<mlir::Value> out;
            for (unsigned k = 0; k < carried; ++k)
              out.push_back(
                  mlir::arith::SelectOp::create(bb, bl, m, next[k], in[k]));
            if (needTaken)
              out.push_back(
                  mlir::arith::OrIOp::create(bb, bl, acc[carried], m));
            return out;
          };

          llvm::SmallVector<mlir::Value> red =
              hlfir::genLoopNestWithReductions(l, b, {extents[axis]}, inits,
                                               body, /*isUnordered=*/false);
          mlir::Value v = red[0];
          if constexpr (!isSum && !isProduct) {
            // Same reconciliation as the total reduction: the seed is an
            // infinity so that an all-infinite slice is not reported as HUGE,
            // and the standard's empty answer is put back afterwards. Here
            // "empty" is per output element, since one slice can be empty
            // only if the contracted extent is zero - but with a mask it is
            // per element in earnest, because a slice can be wholly masked
            // out while its neighbour is not.
            mlir::Value zeroIdx = b.createIntegerConstant(l, idxTy, 0);
            mlir::Value empty = mlir::arith::CmpIOp::create(
                b, l, mlir::arith::CmpIPredicate::eq, extents[axis], zeroIdx);
            if (needTaken) {
              mlir::Value one = b.createIntegerConstant(l, b.getI1Type(), 1);
              empty = mlir::arith::XOrIOp::create(b, l, red.back(), one);
            }
            mlir::Value huge =
                isMax ? genOcta(l, b, 0xFFFFEFFFFFFFFFFFULL, 1)
                      : genOcta(l, b, 0x7FFFEFFFFFFFFFFFULL, 1);
            v = mlir::arith::SelectOp::create(b, l, empty, huge, v);
          }
          return hlfir::Entity{v};
        },
        /*isUnordered=*/true, /*polymorphicMold=*/{}, operation.getType());

    rewriter.replaceOp(operation, elemental.getResult());
    return mlir::success();
  }

  /// MINLOC/MAXLOC over a rank-1 REAL(32) array, as a loop of liboctamath
  /// calls carrying both the best value and its position.
  ///
  /// Without BACK, ties go to the first occurrence, which falls out of
  /// comparing strictly: a later equal element gives code 0 and is not taken.
  /// With BACK they go to the last, so the comparison becomes non-strict and
  /// an equal element displaces the one before it. That is the whole of BACK,
  /// and writing it this way rather than by scanning backwards matters: the
  /// backwards scan changes which element first establishes the running
  /// extremum, and so changes the answer for NaN and for masked elements,
  /// which BACK is not supposed to touch. Because it is a select on a value
  /// rather than a branch, a BACK that is only known at run time costs the
  /// same as a constant one.
  ///
  /// The seed position is 0, and an element is taken when the position is
  /// still 0 and the element is ordered. That gives the standard's answer for
  /// an empty array - zero, since the loop never runs - without a second
  /// reconciliation step, and the same seed gives zero for an array every
  /// element of which is masked out. The ordered gate is what keeps a NaN out:
  /// without it the first element seeds unconditionally, and because every
  /// later comparison against a NaN is unordered, nothing displaces it.
  ///
  /// For rank n the accumulator carries n positions beside the value, and the
  /// result is a rank-1 array of those n subscripts. "First occurrence" is in
  /// array element order, which is what the loop nest already produces:
  /// hlfir::genLoopNestWithReductions builds from the last extent outwards, so
  /// the innermost loop drives the first subscript, and that is column-major.
  /// BACK reverses which of two equal elements wins in that same order, so it
  /// needs nothing extra for rank n - the rank-2 case where the two extrema
  /// sit at (2,1) and (1,2) is the one that would expose a scan written
  /// backwards instead.
  llvm::LogicalResult genOctaMinMaxLoc(OP operation, fir::FirOpBuilder &builder,
                                       mlir::Location loc, hlfir::Entity array,
                                       mlir::Value mask, mlir::Value back,
                                       mlir::PatternRewriter &rewriter) const {
    constexpr bool isMin = std::is_same_v<OP, hlfir::MinlocOp>;
    llvm::SmallVector<mlir::Value> extents =
        hlfir::genExtentsVector(loc, builder, array);
    const unsigned rank = extents.size();

    mlir::Type idxTy = builder.getIndexType();
    mlir::Value zeroIdx = builder.createIntegerConstant(loc, idxTy, 0);
    llvm::SmallVector<mlir::Value> inits{
        isMin ? genOcta(loc, builder, 0x7FFFF00000000000ULL, 0)  // +infinity
              : genOcta(loc, builder, 0xFFFFF00000000000ULL, 0)}; // -infinity
    for (unsigned k = 0; k < rank; ++k)
      inits.push_back(zeroIdx);

    // A scalar mask does not vary with the element, so it is read once here
    // rather than once per iteration. The Entity is built only when there is
    // a mask: hlfir::Entity dereferences the value in its constructor, so
    // wrapping a null one crashes rather than yielding something empty.
    std::optional<hlfir::Entity> maskEntity;
    mlir::Value scalarMask;
    if (mask) {
      maskEntity = hlfir::Entity{mask};
      if (maskEntity->isScalar())
        scalarMask = builder.createConvert(
            loc, builder.getI1Type(),
            hlfir::loadTrivialScalar(loc, builder, *maskEntity));
    }

    auto body = [&](mlir::Location l, fir::FirOpBuilder &b, mlir::ValueRange i,
                    mlir::ValueRange acc) -> llvm::SmallVector<mlir::Value> {
      hlfir::Entity elem = hlfir::loadElementAt(l, b, array, i);
      mlir::Value best = acc[0];
      mlir::Value unordered;
      mlir::Value code = genOctaCmp(l, b, elem, best, &unordered);
      mlir::Value zeroI32 = b.createIntegerConstant(l, b.getI32Type(), 0);
      mlir::Value better = mlir::arith::CmpIOp::create(
          b, l,
          isMin ? mlir::arith::CmpIPredicate::slt
                : mlir::arith::CmpIPredicate::sgt,
          code, zeroI32);
      // BACK makes the comparison non-strict, so an element equal to the
      // running extremum displaces it and the last occurrence survives. It is
      // folded into the predicate rather than reversing the loop, because the
      // loop order also decides which element seeds the accumulator, and BACK
      // must not move that.
      if (back) {
        mlir::Value backBit = b.createConvert(l, b.getI1Type(), back);
        mlir::Value tie = mlir::arith::CmpIOp::create(
            b, l, mlir::arith::CmpIPredicate::eq, code, zeroI32);
        better = mlir::arith::OrIOp::create(
            b, l, better, mlir::arith::AndIOp::create(b, l, backBit, tie));
      }
      // octa_cmp reports unordered separately precisely so that it cannot be
      // ignored by accident: a NaN yields code 0, which would read as "equal"
      // and is not the same thing.
      mlir::Value ordered = mlir::arith::CmpIOp::create(
          b, l, mlir::arith::CmpIPredicate::eq, unordered, zeroI32);
      better = mlir::arith::AndIOp::create(b, l, better, ordered);
      mlir::Value zIdx = b.createIntegerConstant(l, b.getIndexType(), 0);
      mlir::Value nothingYet = mlir::arith::CmpIOp::create(
          b, l, mlir::arith::CmpIPredicate::eq, acc[1], zIdx);
      // Seeding is gated on the element being ordered, so a NaN cannot become
      // the running extremum merely by arriving first. Without this the seed
      // arm takes element one unconditionally, and since every later
      // comparison against a NaN is unordered, nothing ever displaces it:
      // maxloc([NaN,4,7,7]) returned 1 where every other kind returns 3.
      // A NaN is unordered with the +/-infinity seed, so this costs nothing
      // for an array that has none.
      nothingYet = mlir::arith::AndIOp::create(b, l, nothingYet, ordered);
      mlir::Value take = mlir::arith::OrIOp::create(b, l, better, nothingYet);
      // The mask gates the whole decision, including the "nothing yet" arm:
      // a masked-out element must not be taken merely because it came first,
      // which is the difference between MINLOC over the selected elements and
      // MINLOC over the array.
      if (mlir::Value m = scalarMask) {
        take = mlir::arith::AndIOp::create(b, l, take, m);
      } else if (mask) {
        hlfir::Entity mElem = hlfir::loadElementAt(l, b, *maskEntity, i);
        mlir::Value mBit = b.createConvert(l, b.getI1Type(), mElem);
        take = mlir::arith::AndIOp::create(b, l, take, mBit);
      }
      llvm::SmallVector<mlir::Value> next;
      next.push_back(mlir::arith::SelectOp::create(b, l, take, elem, best));
      for (unsigned k = 0; k < rank; ++k)
        next.push_back(
            mlir::arith::SelectOp::create(b, l, take, i[k], acc[k + 1]));
      return next;
    };

    llvm::SmallVector<mlir::Value> reduced = hlfir::genLoopNestWithReductions(
        loc, builder, extents, inits, body, /*isUnordered=*/false);

    mlir::Type resultEleTy = hlfir::getFortranElementType(operation.getType());
    llvm::SmallVector<mlir::Value> positions;
    for (unsigned k = 0; k < rank; ++k)
      positions.push_back(
          builder.createConvert(loc, resultEleTy, reduced[k + 1]));

    // With DIM on a rank-1 array the result is a scalar; otherwise it is a
    // rank-1 array whose length is the rank of the argument.
    if (mlir::isa<hlfir::ExprType>(operation.getType())) {
      mlir::Value n = builder.createIntegerConstant(loc, idxTy, rank);
      mlir::Value shape = builder.genShape(loc, {n});
      hlfir::ElementalOp elemental = hlfir::genElementalOp(
          loc, builder, resultEleTy, shape, /*typeParams=*/{},
          [&](mlir::Location l, fir::FirOpBuilder &b,
              mlir::ValueRange idx) -> hlfir::Entity {
            // A select chain rather than a temporary array: the rank is a
            // compile-time constant and at most 15, and this keeps the result
            // an SSA value.
            mlir::Value res = positions[rank - 1];
            for (unsigned k = rank - 1; k-- > 0;) {
              mlir::Value want =
                  b.createIntegerConstant(l, idx[0].getType(), k + 1);
              mlir::Value is = mlir::arith::CmpIOp::create(
                  b, l, mlir::arith::CmpIPredicate::eq, idx[0], want);
              res = mlir::arith::SelectOp::create(b, l, is, positions[k], res);
            }
            return hlfir::Entity{res};
          },
          /*isUnordered=*/true, /*polymorphicMold=*/{}, operation.getType());
      rewriter.replaceOp(operation, elemental.getResult());
      return mlir::success();
    }
    rewriter.replaceOp(operation, positions[0]);
    return mlir::success();
  }

public:
  llvm::LogicalResult
  matchAndRewrite(OP operation,
                  mlir::PatternRewriter &rewriter) const override {
    std::string opName;
    if constexpr (std::is_same_v<OP, hlfir::SumOp>) {
      opName = "sum";
    } else if constexpr (std::is_same_v<OP, hlfir::ProductOp>) {
      opName = "product";
    } else if constexpr (std::is_same_v<OP, hlfir::MaxvalOp>) {
      opName = "maxval";
    } else if constexpr (std::is_same_v<OP, hlfir::MinvalOp>) {
      opName = "minval";
    } else if constexpr (std::is_same_v<OP, hlfir::MinlocOp>) {
      opName = "minloc";
    } else if constexpr (std::is_same_v<OP, hlfir::MaxlocOp>) {
      opName = "maxloc";
    } else if constexpr (std::is_same_v<OP, hlfir::AnyOp>) {
      opName = "any";
    } else if constexpr (std::is_same_v<OP, hlfir::AllOp>) {
      opName = "all";
    } else {
      return mlir::failure();
    }

    fir::FirOpBuilder builder{rewriter, operation.getOperation()};
    const mlir::Location &loc = operation->getLoc();

    // REAL(32) has no runtime kernel; the reduction is a loop of liboctamath
    // calls, emitted here. Total reductions are covered, with or without a
    // MASK, and so is DIM when it is a compile-time constant naming an axis of
    // the argument. A runtime DIM would mean choosing which axis to contract
    // while the loop nest is being built; that is refused below, before code
    // is generated, rather than reaching the runtime and failing there.
    if constexpr (std::is_same_v<OP, hlfir::SumOp> ||
                  std::is_same_v<OP, hlfir::ProductOp> ||
                  std::is_same_v<OP, hlfir::MaxvalOp> ||
                  std::is_same_v<OP, hlfir::MinvalOp>) {
      hlfir::Entity array{operation.getArray()};
      mlir::Type eleTy = hlfir::getFortranElementType(array.getType());
      mlir::Value mask = operation.getMask();
      bool maskOk = true;
      if (mask) {
        hlfir::Entity m{mask};
        // A box may be an absent optional and a polymorphic mask is not a
        // plain LOGICAL; neither is handled, and neither is guessed at.
        maskOk = !m.isBoxAddressOrValue() && !m.isPolymorphic() &&
                 (m.isScalar() || m.getRank() == array.getRank());
      }
      if (eleTy.isInteger(256) && maskOk) {
        if (!operation.getDim())
          return genOctaReduction(operation, opName, builder, loc, array, mask,
                                  rewriter);
        if (std::optional<llvm::APInt> c =
                fir::getIntIfConstant(operation.getDim())) {
          int64_t d = c->getSExtValue();
          if (d >= 1 && d <= array.getRank()) {
            // A rank-1 argument with DIM=1 is a total reduction whose result
            // the front end already types as a scalar; the elemental form
            // would build a rank-0 array instead.
            if (array.getRank() == 1)
              return genOctaReduction(operation, opName, builder, loc, array,
                                      mask, rewriter);
            return genOctaReductionDim(operation, builder, loc, array, mask,
                                       static_cast<unsigned>(d), rewriter);
          }
        }
        TODO(loc, "REAL(KIND=32) reduction with a non-constant DIM");
      }
    }

    // MINLOC/MAXLOC over REAL(32), any rank, with or without MASK. DIM is
    // accepted only when the argument is rank 1 and DIM is the constant 1,
    // where it is a total reduction and only changes whether the result is a
    // scalar. Any other DIM is a partial reduction whose result is an array of
    // locations, which this does not build; it goes to the runtime and fails
    // there by name.
    //
    // An optional MASK is refused here rather than dereferenced: it may be
    // absent at run time, and the test for that is machinery this lowering
    // does not carry. It too reaches the runtime and fails by name.
    if constexpr (std::is_same_v<OP, hlfir::MinlocOp> ||
                  std::is_same_v<OP, hlfir::MaxlocOp>) {
      hlfir::Entity array{operation.getArray()};
      mlir::Type eleTy = hlfir::getFortranElementType(array.getType());
      bool dimOk = !operation.getDim();
      if (mlir::Value dim = operation.getDim()) {
        if (std::optional<llvm::APInt> c = fir::getIntIfConstant(dim))
          dimOk = c->getSExtValue() == 1 && array.getRank() == 1;
      }
      mlir::Value mask = operation.getMask();
      bool maskOk = true;
      if (mask) {
        hlfir::Entity m{mask};
        // A box may be an absent optional; a polymorphic mask is not a plain
        // LOGICAL. Neither is handled, and neither is guessed at.
        maskOk = !m.isBoxAddressOrValue() && !m.isPolymorphic() &&
                 (m.isScalar() || m.getRank() == array.getRank());
      }
      // BACK is read as a value, so a boxed one - which may be an absent
      // optional - is refused here rather than dereferenced. It reaches the
      // runtime and fails by name, which is where the other unhandled shapes
      // of this intrinsic already go.
      mlir::Value back = operation.getBack();
      bool backOk = true;
      if (back) {
        hlfir::Entity bk{back};
        backOk = !bk.isBoxAddressOrValue() && !bk.isPolymorphic() &&
                 bk.isScalar();
      }
      if (eleTy.isInteger(256) && dimOk && maskOk && backOk)
        return genOctaMinMaxLoc(operation, builder, loc, array, mask, back,
                                rewriter);
    }

    mlir::Type i32 = builder.getI32Type();
    mlir::Type logicalType = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());

    llvm::SmallVector<fir::ExtendedValue, 0> args;

    if constexpr (std::is_same_v<OP, hlfir::SumOp> ||
                  std::is_same_v<OP, hlfir::ProductOp> ||
                  std::is_same_v<OP, hlfir::MaxvalOp> ||
                  std::is_same_v<OP, hlfir::MinvalOp>) {
      args = buildNumericalArgs(operation, i32, logicalType, rewriter, opName);
    } else if constexpr (std::is_same_v<OP, hlfir::MinlocOp> ||
                         std::is_same_v<OP, hlfir::MaxlocOp>) {
      args = buildMinMaxLocArgs(operation, i32, logicalType, rewriter, opName,
                                builder);
    } else {
      args = buildLogicalArgs(operation, i32, logicalType, rewriter, opName);
    }

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(operation.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, opName, scalarResultType, args);

    processReturnValue(operation, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

using SumOpConversion = HlfirReductionIntrinsicConversion<hlfir::SumOp>;

using ProductOpConversion = HlfirReductionIntrinsicConversion<hlfir::ProductOp>;

using MaxvalOpConversion = HlfirReductionIntrinsicConversion<hlfir::MaxvalOp>;

using MinvalOpConversion = HlfirReductionIntrinsicConversion<hlfir::MinvalOp>;

using MinlocOpConversion = HlfirReductionIntrinsicConversion<hlfir::MinlocOp>;

using MaxlocOpConversion = HlfirReductionIntrinsicConversion<hlfir::MaxlocOp>;

using AnyOpConversion = HlfirReductionIntrinsicConversion<hlfir::AnyOp>;

using AllOpConversion = HlfirReductionIntrinsicConversion<hlfir::AllOp>;

struct CountOpConversion : public HlfirIntrinsicConversion<hlfir::CountOp> {
  using HlfirIntrinsicConversion<hlfir::CountOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CountOp count,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, count.getOperation()};
    const mlir::Location &loc = count->getLoc();

    mlir::Type i32 = builder.getI32Type();
    mlir::Type logicalType = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());

    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({count.getMask(), logicalType});
    inArgs.push_back({count.getDim(), i32});
    mlir::Value kind = builder.createIntegerConstant(
        count->getLoc(), i32, getKindForType(count.getType()));
    inArgs.push_back({kind, i32});

    auto *argLowering = fir::getIntrinsicArgumentLowering("count");
    llvm::SmallVector<fir::ExtendedValue, 3> args =
        lowerArguments(count, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType = hlfir::getFortranElementType(count.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "count", scalarResultType, args);

    processReturnValue(count, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

struct MatmulOpConversion : public HlfirIntrinsicConversion<hlfir::MatmulOp> {
  using HlfirIntrinsicConversion<hlfir::MatmulOp>::HlfirIntrinsicConversion;

  /// MATMUL over REAL(32), as an hlfir.elemental whose kernel contracts one
  /// result element with a compensated loop of liboctamath calls.
  ///
  /// Emitted here rather than in SimplifyHLFIRIntrinsics, where the matmul
  /// loop machinery already exists, for the reason the reductions were: that
  /// pass runs only above -O0, and this type had a defect that was exactly a
  /// difference between levels. MATMUL had it too, in its own form - refused
  /// at -O0, and at -O2 the pass rewrote it into arith operations that added
  /// bit patterns, printing 0 for a result of 4.
  ///
  /// The contraction compensates for the same reason SUM and DOT_PRODUCT do:
  /// flang-rt accumulates every other real kind that way, and a REAL(32) that
  /// added naively would be the one kind behaving differently. Each product is
  /// a single octa_mul and so is correctly rounded on its own; what the
  /// compensation repairs is the accumulation across the contraction.
  ///
  /// The three shapes of the intrinsic are handled together because they
  /// differ only in which index is contracted:
  ///
  ///     (n,k) x (k,m) -> (n,m)     r(i,j) = sum_p lhs(i,p) * rhs(p,j)
  ///     (n,k) x (k)   -> (n)       r(i)   = sum_p lhs(i,p) * rhs(p)
  ///     (k)   x (k,m) -> (m)       r(j)   = sum_p lhs(p)   * rhs(p,j)
  mlir::Value genOctaMatmul(fir::FirOpBuilder &builder, mlir::Location loc,
                            hlfir::Entity lhs, hlfir::Entity rhs,
                            mlir::Type resultType) const {
    llvm::SmallVector<mlir::Value> lhsExtents =
        hlfir::genExtentsVector(loc, builder, lhs);
    llvm::SmallVector<mlir::Value> rhsExtents =
        hlfir::genExtentsVector(loc, builder, rhs);
    const bool lhsIsMatrix = lhs.getRank() == 2;
    const bool rhsIsMatrix = rhs.getRank() == 2;

    // The contracted extent is taken from the left operand in every form; the
    // standard requires the two to agree, and taking it from one side rather
    // than the shorter of the two keeps a shape mismatch a shape mismatch
    // instead of quietly truncating the contraction.
    mlir::Value contracted = lhsIsMatrix ? lhsExtents[1] : lhsExtents[0];

    llvm::SmallVector<mlir::Value> resultExtents;
    if (lhsIsMatrix)
      resultExtents.push_back(lhsExtents[0]);
    if (rhsIsMatrix)
      resultExtents.push_back(rhsExtents[1]);
    mlir::Value shape = builder.genShape(loc, resultExtents);

    mlir::Type eleTy = builder.getIntegerType(256);
    auto kernel = [&](mlir::Location l, fir::FirOpBuilder &b,
                      mlir::ValueRange idx) -> hlfir::Entity {
      llvm::SmallVector<mlir::Value> inits{genOcta(l, b, 0, 0),
                                           genOcta(l, b, 0, 0)};
      auto body = [&](mlir::Location bl, fir::FirOpBuilder &bb,
                      mlir::ValueRange p,
                      mlir::ValueRange acc) -> llvm::SmallVector<mlir::Value> {
        llvm::SmallVector<mlir::Value> lhsIdx, rhsIdx;
        if (lhsIsMatrix) {
          lhsIdx = {idx[0], p[0]};
          rhsIdx = rhsIsMatrix ? llvm::SmallVector<mlir::Value>{p[0], idx[1]}
                               : llvm::SmallVector<mlir::Value>{p[0]};
        } else {
          lhsIdx = {p[0]};
          rhsIdx = {p[0], idx[0]};
        }
        hlfir::Entity a = hlfir::loadElementAt(bl, bb, lhs, lhsIdx);
        hlfir::Entity c = hlfir::loadElementAt(bl, bb, rhs, rhsIdx);
        mlir::Value prod = genOctaBinary(bl, bb, "octa_mul", a, c);
        mlir::Value sum = acc[0], corr = acc[1];
        mlir::Value next = genOctaBinary(bl, bb, "octa_sub", prod, corr);
        mlir::Value sumIfNan = genOctaBinary(bl, bb, "octa_add", sum, prod);
        mlir::Value sumOk = genOctaBinary(bl, bb, "octa_add", sum, next);
        mlir::Value diff = genOctaBinary(bl, bb, "octa_sub", sumOk, sum);
        mlir::Value corrOk = genOctaBinary(bl, bb, "octa_sub", diff, next);
        // As in SUM and DOT_PRODUCT: an Inf - Inf makes the correction NaN and
        // would poison a sum that is otherwise well defined, so the
        // compensation is dropped for that step rather than carried.
        mlir::Value unordered;
        (void)genOctaCmp(bl, bb, next, next, &unordered);
        mlir::Value zeroI32 = bb.createIntegerConstant(bl, bb.getI32Type(), 0);
        mlir::Value isNan = mlir::arith::CmpIOp::create(
            bb, bl, mlir::arith::CmpIPredicate::ne, unordered, zeroI32);
        mlir::Value zero = genOcta(bl, bb, 0, 0);
        return {mlir::arith::SelectOp::create(bb, bl, isNan, sumIfNan, sumOk),
                mlir::arith::SelectOp::create(bb, bl, isNan, zero, corrOk)};
      };
      // A zero-length contraction yields zero, which is the seed, so there is
      // nothing to reconcile after the loop.
      mlir::Value dot = hlfir::genLoopNestWithReductions(
          l, b, {contracted}, inits, body, /*isUnordered=*/false)[0];
      return hlfir::Entity{dot};
    };

    hlfir::ElementalOp elemental =
        hlfir::genElementalOp(loc, builder, eleTy, shape, /*typeParams=*/{},
                              kernel, /*isUnordered=*/true,
                              /*polymorphicMold=*/{}, resultType);
    return elemental.getResult();
  }

  llvm::LogicalResult
  matchAndRewrite(hlfir::MatmulOp matmul,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, matmul.getOperation()};
    const mlir::Location &loc = matmul->getLoc();

    mlir::Value lhs = matmul.getLhs();
    mlir::Value rhs = matmul.getRhs();

    // REAL(32) has no runtime kernel: the runtime's matmul is templated over a
    // C++ element type and binary256 has none.
    if (hlfir::getFortranElementType(matmul.getType()).isInteger(256)) {
      mlir::Value result =
          genOctaMatmul(builder, loc, hlfir::Entity{lhs}, hlfir::Entity{rhs},
                        matmul.getType());
      rewriter.replaceOp(matmul, result);
      return mlir::success();
    }

    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("matmul");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(matmul, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(matmul.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "matmul", scalarResultType, args);

    processReturnValue(matmul, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

struct DotProductOpConversion
    : public HlfirIntrinsicConversion<hlfir::DotProductOp> {
  using HlfirIntrinsicConversion<hlfir::DotProductOp>::HlfirIntrinsicConversion;

  /// DOT_PRODUCT over REAL(32), as a compensated loop of liboctamath calls.
  ///
  /// The summation compensates for the same reason SUM does:
  /// flang-rt/lib/runtime/dot-product.cpp accumulates every other real kind
  /// with a compensated sum, and a REAL(32) that added naively would be the one
  /// kind that behaved differently. The width is not an argument against it -
  /// the cancellation Kahan repairs grows with the number of terms, not with
  /// the format.
  ///
  /// Each product is one octa_mul and so is correctly rounded; what the
  /// compensation repairs is the accumulation across terms.
  mlir::Value genOctaDotProduct(fir::FirOpBuilder &builder, mlir::Location loc,
                                hlfir::Entity lhs, hlfir::Entity rhs) const {
    llvm::SmallVector<mlir::Value> extents =
        hlfir::genExtentsVector(loc, builder, lhs);
    llvm::SmallVector<mlir::Value> inits{genOcta(loc, builder, 0, 0),
                                         genOcta(loc, builder, 0, 0)};

    auto body = [&](mlir::Location l, fir::FirOpBuilder &b, mlir::ValueRange idx,
                    mlir::ValueRange acc) -> llvm::SmallVector<mlir::Value> {
      hlfir::Entity a = hlfir::loadElementAt(l, b, lhs, idx);
      hlfir::Entity c = hlfir::loadElementAt(l, b, rhs, idx);
      mlir::Value prod = genOctaBinary(l, b, "octa_mul", a, c);
      mlir::Value sum = acc[0], corr = acc[1];
      mlir::Value next = genOctaBinary(l, b, "octa_sub", prod, corr);
      mlir::Value sumIfNan = genOctaBinary(l, b, "octa_add", sum, prod);
      mlir::Value sumOk = genOctaBinary(l, b, "octa_add", sum, next);
      mlir::Value diff = genOctaBinary(l, b, "octa_sub", sumOk, sum);
      mlir::Value corrOk = genOctaBinary(l, b, "octa_sub", diff, next);
      // An Inf - Inf in the correction makes it NaN and would poison a sum
      // that is otherwise well defined; drop the compensation in that case,
      // exactly as the SUM lowering does.
      mlir::Value unordered;
      (void)genOctaCmp(l, b, next, next, &unordered);
      mlir::Value zeroI32 = b.createIntegerConstant(l, b.getI32Type(), 0);
      mlir::Value isNan = mlir::arith::CmpIOp::create(
          b, l, mlir::arith::CmpIPredicate::ne, unordered, zeroI32);
      mlir::Value zero = genOcta(l, b, 0, 0);
      return {mlir::arith::SelectOp::create(b, l, isNan, sumIfNan, sumOk),
              mlir::arith::SelectOp::create(b, l, isNan, zero, corrOk)};
    };

    // A zero-sized DOT_PRODUCT is zero, which is the seed, so there is nothing
    // to reconcile afterwards.
    return hlfir::genLoopNestWithReductions(loc, builder, extents, inits, body,
                                            /*isUnordered=*/false)[0];
  }

  /// DOT_PRODUCT over COMPLEX(KIND=32).
  ///
  /// The standard conjugates the first argument, so the term is
  /// conj(a) * c = (a_re*c_re + a_im*c_im) + (a_re*c_im - a_im*c_re) i.
  /// The conjugation is therefore in the signs above rather than in a separate
  /// negation, and it is the whole difference between this and the unconjugated
  /// product: getting it wrong yields a perfectly plausible complex number,
  /// which is why it was left out until there was a test whose two candidate
  /// answers differ.
  ///
  /// Each part is accumulated with its own compensation, for the same reason
  /// the real case compensates: flang-rt does so for every other kind, and the
  /// error Kahan repairs grows with the term count, not the format width.
  mlir::Value genOctaComplexDotProduct(fir::FirOpBuilder &builder,
                                       mlir::Location loc, hlfir::Entity lhs,
                                       hlfir::Entity rhs,
                                       mlir::Type resultTy) const {
    llvm::SmallVector<mlir::Value> extents =
        hlfir::genExtentsVector(loc, builder, lhs);
    mlir::Value zero = genOcta(loc, builder, 0, 0);
    // sum_re, corr_re, sum_im, corr_im
    llvm::SmallVector<mlir::Value> inits{zero, zero, zero, zero};

    // One compensated addition: returns the new sum and the new correction.
    auto kahan = [](mlir::Location l, fir::FirOpBuilder &b, mlir::Value sum,
                    mlir::Value corr,
                    mlir::Value term) -> std::pair<mlir::Value, mlir::Value> {
      mlir::Value next = genOctaBinary(l, b, "octa_sub", term, corr);
      mlir::Value sumIfNan = genOctaBinary(l, b, "octa_add", sum, term);
      mlir::Value sumOk = genOctaBinary(l, b, "octa_add", sum, next);
      mlir::Value diff = genOctaBinary(l, b, "octa_sub", sumOk, sum);
      mlir::Value corrOk = genOctaBinary(l, b, "octa_sub", diff, next);
      mlir::Value unordered;
      (void)genOctaCmp(l, b, next, next, &unordered);
      mlir::Value zeroI32 = b.createIntegerConstant(l, b.getI32Type(), 0);
      mlir::Value isNan = mlir::arith::CmpIOp::create(
          b, l, mlir::arith::CmpIPredicate::ne, unordered, zeroI32);
      mlir::Value z = genOcta(l, b, 0, 0);
      return {mlir::arith::SelectOp::create(b, l, isNan, sumIfNan, sumOk),
              mlir::arith::SelectOp::create(b, l, isNan, z, corrOk)};
    };

    auto body = [&](mlir::Location l, fir::FirOpBuilder &b, mlir::ValueRange idx,
                    mlir::ValueRange acc) -> llvm::SmallVector<mlir::Value> {
      hlfir::Entity a = hlfir::loadElementAt(l, b, lhs, idx);
      hlfir::Entity c = hlfir::loadElementAt(l, b, rhs, idx);
      fir::factory::Complex cx{b, l};
      mlir::Value aRe = cx.extractComplexPart(a, /*isImagPart=*/false);
      mlir::Value aIm = cx.extractComplexPart(a, /*isImagPart=*/true);
      mlir::Value cRe = cx.extractComplexPart(c, /*isImagPart=*/false);
      mlir::Value cIm = cx.extractComplexPart(c, /*isImagPart=*/true);

      mlir::Value reRe = genOctaBinary(l, b, "octa_mul", aRe, cRe);
      mlir::Value imIm = genOctaBinary(l, b, "octa_mul", aIm, cIm);
      mlir::Value reIm = genOctaBinary(l, b, "octa_mul", aRe, cIm);
      mlir::Value imRe = genOctaBinary(l, b, "octa_mul", aIm, cRe);
      // conj(a)*c: the imaginary parts add in the real term and subtract in
      // the imaginary one. Reversing either sign is the silent error.
      mlir::Value termRe = genOctaBinary(l, b, "octa_add", reRe, imIm);
      mlir::Value termIm = genOctaBinary(l, b, "octa_sub", reIm, imRe);

      auto [sumRe, corrRe] = kahan(l, b, acc[0], acc[1], termRe);
      auto [sumIm, corrIm] = kahan(l, b, acc[2], acc[3], termIm);
      return {sumRe, corrRe, sumIm, corrIm};
    };

    llvm::SmallVector<mlir::Value> reduced = hlfir::genLoopNestWithReductions(
        loc, builder, extents, inits, body, /*isUnordered=*/false);
    return fir::factory::Complex{builder, loc}.createComplex(
        resultTy, reduced[0], reduced[2]);
  }

  llvm::LogicalResult
  matchAndRewrite(hlfir::DotProductOp dotProduct,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, dotProduct.getOperation()};
    const mlir::Location &loc = dotProduct->getLoc();

    mlir::Value lhs = dotProduct.getLhs();
    mlir::Value rhs = dotProduct.getRhs();

    // REAL(32) and COMPLEX(KIND=32) have no runtime kernel.
    mlir::Type resultTy = dotProduct.getType();
    if (hlfir::getFortranElementType(resultTy).isInteger(256)) {
      mlir::Value result = genOctaDotProduct(builder, loc, hlfir::Entity{lhs},
                                             hlfir::Entity{rhs});
      rewriter.replaceOp(dotProduct, result);
      return mlir::success();
    }
    if (fir::isa_octuple_complex(hlfir::getFortranElementType(resultTy))) {
      mlir::Value result = genOctaComplexDotProduct(
          builder, loc, hlfir::Entity{lhs}, hlfir::Entity{rhs},
          hlfir::getFortranElementType(resultTy));
      rewriter.replaceOp(dotProduct, result);
      return mlir::success();
    }

    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("dot_product");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(dotProduct, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(dotProduct.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "dot_product", scalarResultType, args);

    processReturnValue(dotProduct, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class TransposeOpConversion
    : public HlfirIntrinsicConversion<hlfir::TransposeOp> {
  using HlfirIntrinsicConversion<hlfir::TransposeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::TransposeOp transpose,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, transpose.getOperation()};
    const mlir::Location &loc = transpose->getLoc();

    mlir::Value arg = transpose.getArray();
    llvm::SmallVector<IntrinsicArgument, 1> inArgs;
    inArgs.push_back({arg, arg.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("transpose");
    llvm::SmallVector<fir::ExtendedValue, 1> args =
        lowerArguments(transpose, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(transpose.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "transpose", scalarResultType, args);

    processReturnValue(transpose, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

struct MatmulTransposeOpConversion
    : public HlfirIntrinsicConversion<hlfir::MatmulTransposeOp> {
  using HlfirIntrinsicConversion<
      hlfir::MatmulTransposeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::MatmulTransposeOp multranspose,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, multranspose.getOperation()};
    const mlir::Location &loc = multranspose->getLoc();

    mlir::Value lhs = multranspose.getLhs();
    mlir::Value rhs = multranspose.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("matmul");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(multranspose, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(multranspose.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "matmul_transpose", scalarResultType, args);

    processReturnValue(multranspose, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

// A converter for hlfir.cshift and hlfir.eoshift.
template <typename T>
class ArrayShiftOpConversion : public HlfirIntrinsicConversion<T> {
  using HlfirIntrinsicConversion<T>::HlfirIntrinsicConversion;
  using HlfirIntrinsicConversion<T>::lowerArguments;
  using HlfirIntrinsicConversion<T>::processReturnValue;
  using typename HlfirIntrinsicConversion<T>::IntrinsicArgument;

  llvm::LogicalResult
  matchAndRewrite(T op, mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, op.getOperation()};
    const mlir::Location &loc = op->getLoc();

    llvm::SmallVector<IntrinsicArgument, 4> inArgs;
    llvm::StringRef intrinsicName{[]() {
      if constexpr (std::is_same_v<T, hlfir::EOShiftOp>)
        return "eoshift";
      else if constexpr (std::is_same_v<T, hlfir::CShiftOp>)
        return "cshift";
      else
        llvm_unreachable("unsupported array shift");
    }()};

    mlir::Value array = op.getArray();
    inArgs.push_back({array, array.getType()});
    mlir::Value shift = op.getShift();
    inArgs.push_back({shift, shift.getType()});
    if constexpr (std::is_same_v<T, hlfir::EOShiftOp>) {
      mlir::Value boundary = op.getBoundary();
      inArgs.push_back({boundary, boundary ? boundary.getType() : nullptr});
    }
    inArgs.push_back({op.getDim(), builder.getI32Type()});

    auto *argLowering = fir::getIntrinsicArgumentLowering(intrinsicName);
    llvm::SmallVector<fir::ExtendedValue, 3> args =
        lowerArguments(op, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType = hlfir::getFortranElementType(op.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, intrinsicName, scalarResultType, args);

    processReturnValue(op, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class ReshapeOpConversion : public HlfirIntrinsicConversion<hlfir::ReshapeOp> {
  using HlfirIntrinsicConversion<hlfir::ReshapeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ReshapeOp reshape,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, reshape.getOperation()};
    const mlir::Location &loc = reshape->getLoc();

    llvm::SmallVector<IntrinsicArgument, 4> inArgs;
    mlir::Value array = reshape.getArray();
    inArgs.push_back({array, array.getType()});
    mlir::Value shape = reshape.getShape();
    inArgs.push_back({shape, shape.getType()});
    mlir::Type noneType = builder.getNoneType();
    mlir::Value pad = reshape.getPad();
    inArgs.push_back({pad, pad ? pad.getType() : noneType});
    mlir::Value order = reshape.getOrder();
    inArgs.push_back({order, order ? order.getType() : noneType});

    auto *argLowering = fir::getIntrinsicArgumentLowering("reshape");
    llvm::SmallVector<fir::ExtendedValue, 4> args =
        lowerArguments(reshape, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(reshape.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "reshape", scalarResultType, args);

    processReturnValue(reshape, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class CmpCharOpConversion : public HlfirIntrinsicConversion<hlfir::CmpCharOp> {
  using HlfirIntrinsicConversion<hlfir::CmpCharOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CmpCharOp cmp,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, cmp.getOperation()};
    const mlir::Location &loc = cmp->getLoc();
    hlfir::Entity lhs{cmp.getLchr()};
    hlfir::Entity rhs{cmp.getRchr()};

    auto [lhsExv, lhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, lhs);
    auto [rhsExv, rhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, rhs);

    auto resultVal = fir::runtime::genCharCompare(
        builder, loc, cmp.getPredicate(), lhsExv, rhsExv);
    if (lhsCleanUp || rhsCleanUp) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(cmp);
      if (lhsCleanUp)
        (*lhsCleanUp)();
      if (rhsCleanUp)
        (*rhsCleanUp)();
    }
    auto resultEntity = hlfir::EntityWithAttributes{resultVal};

    processReturnValue(cmp, resultEntity, /*mustBeFreed=*/false, builder,
                       rewriter);
    return mlir::success();
  }
};

class CharTrimOpConversion
    : public HlfirIntrinsicConversion<hlfir::CharTrimOp> {
  using HlfirIntrinsicConversion<hlfir::CharTrimOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CharTrimOp trim,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, trim.getOperation()};
    const mlir::Location &loc = trim->getLoc();

    llvm::SmallVector<IntrinsicArgument, 1> inArgs;
    mlir::Value chr = trim.getChr();
    inArgs.push_back({chr, chr.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("trim");
    llvm::SmallVector<fir::ExtendedValue, 1> args =
        lowerArguments(trim, inArgs, rewriter, argLowering);

    mlir::Type resultType = hlfir::getFortranElementType(trim.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "trim", resultType, args);

    processReturnValue(trim, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class IndexOpConversion : public HlfirIntrinsicConversion<hlfir::IndexOp> {
  using HlfirIntrinsicConversion<hlfir::IndexOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::IndexOp op,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, op.getOperation()};
    const mlir::Location &loc = op->getLoc();
    hlfir::Entity substr{op.getSubstr()};
    hlfir::Entity str{op.getStr()};

    auto [substrExv, substrCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, substr);
    auto [strExv, strCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, str);

    mlir::Value back = op.getBack();
    if (!back)
      back = builder.createBool(loc, false);

    mlir::Value result =
        fir::runtime::genIndex(builder, loc, strExv, substrExv, back);
    result = builder.createConvert(loc, op.getType(), result);
    if (strCleanUp || substrCleanUp) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(op);
      if (strCleanUp)
        (*strCleanUp)();
      if (substrCleanUp)
        (*substrCleanUp)();
    }
    auto resultEntity = hlfir::EntityWithAttributes{result};

    processReturnValue(op, resultEntity, /*mustBeFreed=*/false, builder,
                       rewriter);
    return mlir::success();
  }
};

class LowerHLFIRIntrinsics
    : public hlfir::impl::LowerHLFIRIntrinsicsBase<LowerHLFIRIntrinsics> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<
        MatmulOpConversion, MatmulTransposeOpConversion, AllOpConversion,
        AnyOpConversion, SumOpConversion, ProductOpConversion,
        TransposeOpConversion, CountOpConversion, DotProductOpConversion,
        MaxvalOpConversion, MinvalOpConversion, MinlocOpConversion,
        MaxlocOpConversion, ArrayShiftOpConversion<hlfir::CShiftOp>,
        ArrayShiftOpConversion<hlfir::EOShiftOp>, ReshapeOpConversion,
        CmpCharOpConversion, CharTrimOpConversion, IndexOpConversion>(context);

    // While conceptually this pass is performing dialect conversion, we use
    // pattern rewrites here instead of dialect conversion because this pass
    // looses array bounds from some of the expressions e.g.
    // !hlfir.expr<2xi32> -> !hlfir.expr<?xi32>
    // MLIR thinks this is a different type so dialect conversion fails.
    // Pattern rewriting only requires that the resulting IR is still valid
    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR intrinsic lowering");
      signalPassFailure();
    }
  }
};
} // namespace
