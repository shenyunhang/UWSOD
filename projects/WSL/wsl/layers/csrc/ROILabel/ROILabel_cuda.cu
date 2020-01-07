#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

namespace wsl {

std::tuple<at::Tensor, at::Tensor, at::Tensor> ROILabel_forward_cuda(
    const at::Tensor& S,
    const at::Tensor& U,
    const at::Tensor& L,
    const at::Tensor& CW,
    const at::Tensor& P) {
  AT_ASSERTM(S.is_cuda(), "S must be a CUDA tensor");
  AT_ASSERTM(U.is_cuda(), "U must be a CUDA tensor");
  AT_ASSERTM(L.is_cuda(), "L must be a CUDA tensor");
  AT_ASSERTM(CW.is_cuda(), "CW must be a CUDA tensor");
  AT_ASSERTM(P.is_cuda(), "P must be a CUDA tensor");

  at::TensorArg S_t{S, "S", 1}, U_t{U, "U", 2}, CW_t{CW, "CW", 4};

  at::CheckedFrom c = "ROILabel_forward_cuda";
  at::checkAllSameGPU(c, {S_t, U_t, CW_t});
  at::checkAllSameType(c, {S_t, U_t, CW_t});

  at::cuda::CUDAGuard device_guard(S.device());

  AT_ASSERT(S.dim() == 2);
  AT_ASSERT(U.dim() == 2);
  AT_ASSERT(L.dim() == 2);
  AT_ASSERT(S.size(0) == U.size(0));
  AT_ASSERT((S.size(1) == L.size(1)) || (S.size(1) == L.size(1) + 1));
  AT_ASSERT(U.size(0) == U.size(1));
  AT_ASSERT(L.size(0) == 1);

  const int num_roi = S.size(0);
  const int num_class_s = S.size(1);
  const int num_class = L.size(1);
  const int offset_class = num_class_s - num_class;

  at::Tensor RL = at::zeros({num_roi}, S.options().dtype(at::kInt));
  at::Tensor RW = at::zeros({num_roi}, S.options());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(RL, RW, P);
}

} // namespace wsl
