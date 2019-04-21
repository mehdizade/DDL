/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except on compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/shape_inference.h"
//#include "tensorflow/core/util/padding.h"


using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("LassoFistaLayer9")
    .Input("dict: float")
    .Input("input: float")
    .Input("lambda1: float")
    .Input("a: float")
    .Input("ksize: int32")
    .Input("strides: int32")
    .Input("rates: int32")
    .Input("pad: int32")
    .Input("dims: int32")
    .Output("output: float")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.
output: A Tensor.
  output = input + 1
)doc");


void LassoFistaKernelLauncher(const float* in, const int N, float* out);
//void LassoFistaKernelLauncherFloat(const float* in, const int N, float* out);
//void LassoFistaKernelLauncherDouble(const double* in, const int N, double* out);


class LassoFistaOpFloat : public OpKernel {
 public:
  explicit LassoFistaOpFloat(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(3);
    auto input = input_tensor.flat<float>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0,  input_tensor.shape(), &output_tensor));
    auto output = output_tensor-> flat<float>();

    const int N = input.size();
    LassoFistaKernelLauncher(input.data(), N, output.data());

  }
};

REGISTER_KERNEL_BUILDER(
        Name("LassoFistaLayer9")
        .Device(DEVICE_GPU),
        LassoFistaOpFloat);
