// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: my_relu_fold
tt.func @my_relu_fold(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  // my_relu(my_relu(x)) 应该被折叠为 my_relu(x)
  // CHECK: tt.my_relu
  // CHECK-NOT: tt.my_relu
  %0 = tt.my_relu %arg0 : tensor<128xf32>
  %1 = tt.my_relu %0 : tensor<128xf32>
  tt.return %1 : tensor<128xf32>
}


// -----

// CHECK-LABEL: my_relu_const_nonneg
tt.func @my_relu_const_nonneg() -> tensor<4xf32> {
  // 全非负常量，relu 应被消除
  // CHECK: arith.constant
  // CHECK-NOT: tt.my_relu
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = tt.my_relu %0 : tensor<4xf32>
  tt.return %1 : tensor<4xf32>
}
