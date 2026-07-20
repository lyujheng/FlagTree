module {
  tt.func @test_relu() {
    %cst = arith.constant dense<1.0> : tensor<128xf32>
    %result = tt.my_relu %cst : tensor<128xf32>
    tt.return
  }
}
