#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @test(%arg0: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    %cst = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    %result = arith.addf %arg0, %cst : tensor<128xf32, #blocked>
    tt.return %result : tensor<128xf32, #blocked>
  }
}