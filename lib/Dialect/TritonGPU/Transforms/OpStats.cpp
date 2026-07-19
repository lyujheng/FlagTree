
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"


namespace mlir{
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPSTATS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct OpStatsPass : impl::TritonGPUOpStatsBase<OpStatsPass> {
    using Base::Base;

    void runOnOperation() override {
        ModuleOp moduleOp = getOperation();
        llvm::StringMap<unsigned> opCount;
        
        moduleOp.walk([&](Operation *op){
            ++opCount[op->getName().getStringRef()];
        });

        for (const auto& entry : opCount) {
            llvm::errs() << entry.getKey() << " : " << entry.getValue() << "\n";
        }
    }
}; // struct OpStatsPass
} // namespace gpu
} // namespace triton
} // namespace mlir