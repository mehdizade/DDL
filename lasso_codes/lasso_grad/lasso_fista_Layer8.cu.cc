/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/



#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/kernels/extract_image_patches_op.h"
//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


__global__ void LassoFistaKernel(const float* in, const int N, float* out) {
   // out=in;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    out[i] = in[i] ;  
  }
}


void LassoFistaKernelLauncher(const float* in, const int N, float* out) {
  LassoFistaKernel<<<32, 256>>>(in, N, out);
}

//void LassoFistaKernelLaunchert(const float* in, const int N, float* out) {
//  LassoFistaKernel<float><<<32, 256>>>(in, N, out);
//  LassoFistaKernel<T>(in, N, out);
//}

//void LassoFistaKernelLauncherDouble(const double* in, const int N, double* out) {
//  LassoFistaKernel<double><<<32, 256>>>(in, N, out);
//  LassoFistaKernel<T>(in, N, out);
//}

#endif
