#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

namespace wsl {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ROIMerge_forward_cuda(
    const at::Tensor& S,
    const at::Tensor& J,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& P) {
  AT_ASSERTM(S.is_cuda(), "S must be a CUDA tensor");
  AT_ASSERTM(J.is_cuda(), "J must be a CUDA tensor");
  AT_ASSERTM(C.is_cuda(), "C must be a CUDA tensor");
  AT_ASSERTM(D.is_cuda(), "D must be a CUDA tensor");

  at::TensorArg S_t{S, "S", 1}, J_t{J, "J", 2}, C_t{C, "C", 3}, D_t{D, "D", 4};

  at::CheckedFrom c = "ROIMerge_forward_cuda";
  at::checkAllSameGPU(c, {S_t, J_t, C_t, D_t});
  at::checkAllSameType(c, {S_t, J_t, C_t, D_t});

  at::cuda::CUDAGuard device_guard(C.device());

  AT_ASSERT(S.dim() == 2);
  AT_ASSERT(J.dim() == 2);
  AT_ASSERT(C.dim() == 2);
  AT_ASSERT(D.dim() == 2);
  AT_ASSERT(S.size(0) == J.size(0));
  AT_ASSERT(S.size(0) == C.size(0));
  AT_ASSERT(S.size(0) == D.size(0));
  AT_ASSERT(S.size(1) == 1);
  AT_ASSERT(J.size(0) == J.size(1));
  AT_ASSERT(C.size(1) == D.size(1));

  const int num_roi = C.size(0);
  const int num_class = C.size(1);

  const int num_id = 1;

  at::Tensor I = at::zeros({num_roi}, S.options().dtype(at::kInt));
  I.fill_(-1);

  at::Tensor MC = at::zeros({num_id, num_class}, S.options());
  at::Tensor MD = at::zeros({num_id, num_class}, S.options());
  at::Tensor IC = at::zeros({num_roi}, S.options().dtype(at::kInt));
  IC.fill_(0);

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(MC, MD, I, IC, P);
}

std::tuple<at::Tensor, at::Tensor> ROIMerge_backward_cuda(
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& GMC,
    const at::Tensor& GMD,
    const at::Tensor& I,
    const at::Tensor& IC) {
  // Check if input tensors are CUDA tensors
  AT_ASSERTM(C.is_cuda(), "C must be a CUDA tensor");
  AT_ASSERTM(D.is_cuda(), "D must be a CUDA tensor");
  AT_ASSERTM(GMC.is_cuda(), "GMC must be a CUDA tensor");
  AT_ASSERTM(GMD.is_cuda(), "GMD must be a CUDA tensor");
  AT_ASSERTM(I.is_cuda(), "I must be a CUDA tensor");
  AT_ASSERTM(IC.is_cuda(), "IC must be a CUDA tensor");

  at::TensorArg C_t{C, "C", 1}, D_t{D, "D", 2}, GMC_t{GMC, "GMC", 2},
      GMD_t{GMD, "GMD", 3}, I_t{I, "I", 4}, IC_t{IC, "IC", 5};

  at::CheckedFrom c = "ROIMerge_backward_cuda";
  at::checkAllSameGPU(c, {C_t, D_t, GMC_t, GMD_t});
  at::checkAllSameGPU(c, {I_t, IC_t});
  at::checkAllSameType(c, {C_t, D_t, GMC_t, GMD_t});
  at::checkAllSameType(c, {I_t, IC_t});

  at::cuda::CUDAGuard device_guard(C.device());

  const int num_roi = C.size(0);
  const int num_class = C.size(1);

  at::Tensor GC = at::zeros({num_roi, num_class}, C.options());
  at::Tensor GD = at::zeros({num_roi, num_class}, D.options());

  return std::make_tuple(GC, GD);
}

} // namespace wsl
