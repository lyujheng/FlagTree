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

