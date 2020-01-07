#pragma once
#include <torch/types.h>

namespace wsl {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ROIMerge_forward_cpu(
    const at::Tensor& S,
    const at::Tensor& J,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& P);

std::tuple<at::Tensor, at::Tensor> ROIMerge_backward_cpu(
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& GMC,
    const at::Tensor& GMD,
    const at::Tensor& I,
    const at::Tensor& IC);

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ROIMerge_forward_cuda(
    const at::Tensor& S,
    const at::Tensor& J,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& P);

std::tuple<at::Tensor, at::Tensor> ROIMerge_backward_cuda(
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& GMC,
    const at::Tensor& GMD,
    const at::Tensor& I,
    const at::Tensor& IC);
#endif

// Interface for Python
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ROIMerge_forward(
    const at::Tensor& S,
    const at::Tensor& J,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& P) {
  if (S.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    AT_ERROR("Not compiled with GPU support");
    return ROIMerge_forward_cuda(S, J, C, D, P);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIMerge_forward_cpu(S, J, C, D, P);
}

inline std::tuple<at::Tensor, at::Tensor> ROIMerge_backward(
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& GMC,
    const at::Tensor& GMD,
    const at::Tensor& I,
    const at::Tensor& IC) {
  if (C.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    AT_ERROR("Not compiled with GPU support");
    return ROIMerge_backward_cuda(C, D, GMC, GMD, I, IC);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIMerge_backward_cpu(C, D, GMC, GMD, I, IC);
}

} // namespace wsl
