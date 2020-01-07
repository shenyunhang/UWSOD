#pragma once
#include <torch/types.h>

namespace wsl {

std::tuple<at::Tensor, at::Tensor, at::Tensor> ROILabel_forward_cpu(
    const at::Tensor& S,
    const at::Tensor& U,
    const at::Tensor& L,
    const at::Tensor& CW,
    const at::Tensor& P);

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor, at::Tensor> ROILabel_forward_cuda(
    const at::Tensor& S,
    const at::Tensor& U,
    const at::Tensor& L,
    const at::Tensor& CW,
    const at::Tensor& P);
#endif

// Interface for Python
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> ROILabel_forward(
    const at::Tensor& S,
    const at::Tensor& U,
    const at::Tensor& L,
    const at::Tensor& CW,
    const at::Tensor& P) {
  if (S.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    AT_ERROR("Not compiled with GPU support");
    return ROILabel_forward_cuda(S, U, L, CW, P);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROILabel_forward_cpu(S, U, L, CW, P);
}

} // namespace wsl
