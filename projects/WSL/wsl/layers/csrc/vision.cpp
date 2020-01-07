// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/extension.h>
#include "ROILabel/ROILabel.h"
#include "ROILoopPool/ROILoopPool.h"
#include "ROIMerge/ROIMerge.h"
#include "crf/crf.h"
#include "csc/csc.h"
#include "pcl_loss/pcl_loss.h"

namespace wsl {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pcl_loss_forward", &pcl_loss_forward, "pcl_loss_forward");
  m.def("pcl_loss_backward", &pcl_loss_backward, "pcl_loss_backward");

  m.def("csc_forward", &csc_forward, "csc_forward");

  m.def("crf_forward", &crf_forward, "crf_forward");

  m.def("roi_loop_pool_forward", &ROILoopPool_forward, "ROILoopPool_forward");
  m.def(
      "roi_loop_pool_backward", &ROILoopPool_backward, "ROILoopPool_backward");

  m.def("roi_merge_forward", &ROIMerge_forward, "ROIMerge_forward");
  m.def("roi_merge_backward", &ROIMerge_backward, "ROIMerge_backward");

  m.def("roi_label_forward", &ROILabel_forward, "ROILabel_forward");
}

} // namespace wsl
