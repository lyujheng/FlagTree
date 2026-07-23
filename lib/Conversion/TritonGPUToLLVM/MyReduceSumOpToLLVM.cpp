#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct MyReduceSumOpConversion
    : public ConvertOpToLLVMPattern<triton::MyReduceSumOp> {
public:
  MyReduceSumOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::MyReduceSumOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::MyReduceSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // 1. 线程内规约
    auto srcValues = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    Value acc = srcValues[0];
    for (unsigned i = 1; i < srcValues.size(); ++i) {
      acc = b.fadd(acc, srcValues[i]);
    }

    // 2. Warp 内 shuffle 规约
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
      Value shfl = targetInfo.shuffleXor(rewriter, loc, acc, offset);
      acc = b.fadd(acc, shfl);
    }

    // 3. 跨 Warp 规约
    acc = reduceAcrossWarps(loc, b, rewriter, op, acc);

    // 4. 输出结果
    rewriter.replaceOp(op, acc);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  Value reduceAcrossWarps(Location loc, TritonLLVMOpBuilder &b,
                          ConversionPatternRewriter &rewriter,
                          triton::MyReduceSumOp op, Value acc) const {
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(32);
    Value laneId = b.urem(threadId, warpSize);
    Value warpId = b.udiv(threadId, warpSize);

    int numWarps = triton::gpu::lookupNumWarps(op.getOperation());
    auto elemTy = cast<RankedTensorType>(op.getSrc().getType()).getElementType();
    Value smemBase = mlir::LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());

    // 每个 warp 的 lane 0 写入 shared memory
    Value isLaneZero = b.icmp_eq(laneId, b.i32_val(0));
    Value writePtr = b.gep(smemBase.getType(), elemTy, smemBase, warpId);
    targetInfo.storeShared(rewriter, loc, writePtr, acc, isLaneZero);

    b.barrier();

    // 第一个 warp 读取所有 warp 结果
    Value isFirstWarp = b.icmp_eq(warpId, b.i32_val(0));
    Value canRead = b.and_(isFirstWarp, b.icmp_slt(laneId, b.i32_val(numWarps)));
    Value readPtr = b.gep(smemBase.getType(), elemTy, smemBase, laneId);
    acc = targetInfo.loadShared(rewriter, loc, readPtr, elemTy, canRead);

    // canRead 为 false 的 lane 赋 0，避免累加脏数据
    Value zero = rewriter.create<LLVM::ConstantOp>(loc, elemTy, rewriter.getFloatAttr(elemTy, 0.0));
    acc = b.select(canRead, acc, zero);

    // 第一个 warp 内 shuffle 规约
    unsigned reduceSize = llvm::PowerOf2Ceil(numWarps);
    for (unsigned offset = reduceSize / 2; offset > 0; offset >>= 1) {
      Value shfl = targetInfo.shuffleXor(rewriter, loc, acc, offset);
      acc = b.fadd(acc, shfl);
    }

    return acc;
  }
};
} // namespace

void mlir::triton::populateMyReduceSumOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<MyReduceSumOpConversion>(typeConverter, targetInfo, benefit);
}
