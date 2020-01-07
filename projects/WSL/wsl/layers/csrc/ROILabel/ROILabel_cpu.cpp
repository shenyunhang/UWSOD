#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>

#include <algorithm> // std::random_shuffle
#include <cfloat>
#include <cstdlib> // std::rand, std::srand
#include <ctime> // std::time
#include <functional>
#include <vector> // std::vector

#include "ROILabel.h"

namespace wsl {

std::tuple<at::Tensor, at::Tensor, at::Tensor> ROILabel_forward_cpu(
    const at::Tensor& S,
    const at::Tensor& U,
    const at::Tensor& L,
    const at::Tensor& CW,
    const at::Tensor& P) {
  AT_ASSERTM(S.device().is_cpu(), "S must be a CPU tensor");
  AT_ASSERTM(U.device().is_cpu(), "U must be a CPU tensor");
  AT_ASSERTM(L.device().is_cpu(), "L must be a CPU tensor");
  AT_ASSERTM(CW.device().is_cpu(), "CW must be a CPU tensor");
  AT_ASSERTM(P.device().is_cpu(), "P must be a CPU tensor");

  at::TensorArg S_t{S, "S", 1}, U_t{U, "U", 2}, CW_t{CW, "CW", 4};

  at::CheckedFrom c = "ROILabel_forward_cpu";
  at::checkAllSameType(c, {S_t, U_t, CW_t});

  AT_ASSERT(S.dim() == 2);
  AT_ASSERT(U.dim() == 2);
  AT_ASSERT(L.dim() == 2);
  AT_ASSERT(S.size(0) == U.size(0));
  AT_ASSERT((S.size(1) == L.size(1)) || (S.size(1) == L.size(1) + 1));
  AT_ASSERT(U.size(0) == U.size(1));
  AT_ASSERT(L.size(0) == 1);

  AT_ASSERT(P.dim() == 1);
  AT_ASSERT(P.size(0) == 14);
  auto Pdata = P.contiguous().data_ptr<float>();
  float fg_thresh_ = Pdata[0];
  float bg_thresh_hi_ = Pdata[1];
  float bg_thresh_lo_ = Pdata[2];
  int num_pos_ = int(Pdata[3]);
  int num_neg_ = int(Pdata[4]);
  int top_k_ = int(Pdata[5]);
  int debug_info_ = int(Pdata[6]);
  int uuid_ = int(Pdata[7]);
  int display_ = int(Pdata[8]);
  int cur_iter_ = int(Pdata[9]);
  int acc_fg_rois_ = int(Pdata[10]);
  int acc_bg_rois_ = int(Pdata[11]);
  float acc_fg_weight_ = Pdata[12];
  float acc_bg_weight_ = Pdata[13];

  const int num_roi = S.size(0);
  const int num_class_s = S.size(1);
  const int num_class = L.size(1);
  // const int offset_class = num_class_s - num_class;
  const int offset_class = 0;

  at::Tensor RL = at::zeros({num_roi}, S.options().dtype(at::kInt));
  at::Tensor RW = at::zeros({num_roi}, S.options());

  auto S_ = S.contiguous(), U_ = U.contiguous(), CW_ = CW.contiguous(),
       L_ = L.contiguous();

  const float* Sdata = S_.data_ptr<float>();
  const float* Udata = U_.data_ptr<float>();
  const float* Ldata = L_.data_ptr<float>();
  const float* CWdata = CW_.data_ptr<float>();

  auto RL_ = RL.contiguous(), RW_ = RW.contiguous();
  int* RLdata = RL_.data_ptr<int>();
  float* RWdata = RW_.data_ptr<float>();

  std::vector<int> highest_n;
  std::vector<int> highest_c;
  std::vector<float> highest_p;

  for (int c = 0; c < num_class; c++) {
    if (Ldata[c] == 1) {
    } else {
      continue;
    }

    for (int k = 0; k < top_k_; k++) {
      float max_pred = -FLT_MAX;
      int max_idx = -1;
      for (int n = 0; n < num_roi; n++) {
        if (max_pred < Sdata[n * num_class_s + c + offset_class]) {
          if (std::find(highest_n.begin(), highest_n.end(), n) !=
              highest_n.end()) {
          } else {
            max_pred = Sdata[n * num_class_s + c + offset_class];
            max_idx = n;
          }
        }
      }

      highest_n.push_back(max_idx);
      highest_c.push_back(c);
      highest_p.push_back(max_pred);
    }
  }

  std::srand(unsigned(std::time(0)));
  std::vector<int> myvector;

  // set some values:
  for (int n = 0; n < num_roi; n++) {
    myvector.push_back(n); // 1 2 3 4 5 6 7 8 9
  }
  // using built-in random generator:
  std::random_shuffle(myvector.begin(), myvector.end());

  int num_pos = 0;
  int num_neg = 0;

  // for (int n = 0; n < num_roi; n++) {
  for (std::vector<int>::iterator it = myvector.begin(); it != myvector.end();
       ++it) {
    int n = *it;
    float max_iou = -FLT_MAX;
    int max_idx = -1;
    for (int i = 0; i < highest_n.size(); i++) {
      int g = highest_n[i];
      if (max_iou < Udata[n * num_roi + g]) {
        max_iou = Udata[n * num_roi + g];
        max_idx = i;
      }
    }

    int assign_n = highest_n[max_idx];
    int assign_c = highest_c[max_idx];
    float assign_w = CWdata ? CWdata[assign_c] : highest_p[max_idx];

    if (max_iou >= fg_thresh_ && num_pos <= num_pos_) {
      assign_c = assign_c;
      num_pos++;
      acc_fg_rois_++;
      acc_fg_weight_ += assign_w;
    } else if (
        max_iou >= bg_thresh_lo_ && max_iou < bg_thresh_hi_ &&
        num_neg <= num_neg_) {
      assign_c = num_class;
      num_neg++;
      acc_bg_rois_++;
      acc_bg_weight_ += assign_w;
    } else {
      assign_c = assign_c;
      assign_w = 0;
    }

    RLdata[n] = assign_c;
    RWdata[n] = assign_w;
    // RWdata[n] = 1;
  }

  cur_iter_++;
  if (cur_iter_ % display_ == 0) {
    printf(
        "RoILabel %d\tfg_rois: %d\tbg_rois: %d\tfg_weight: %f\tbg_weight: %f\n",
        uuid_,
        acc_fg_rois_ / display_,
        acc_bg_rois_ / display_,
        acc_fg_weight_ / acc_fg_rois_,
        acc_bg_weight_ / acc_bg_rois_);
    acc_fg_rois_ = 0;
    acc_bg_rois_ = 0;
    acc_fg_weight_ = 0;
    acc_bg_weight_ = 0;
  }

  Pdata[0] = fg_thresh_;
  Pdata[1] = bg_thresh_hi_;
  Pdata[2] = bg_thresh_lo_;
  Pdata[3] = num_pos_;
  Pdata[4] = num_neg_;
  Pdata[5] = top_k_;
  Pdata[6] = debug_info_;
  Pdata[7] = uuid_;
  Pdata[8] = display_;
  Pdata[9] = cur_iter_;
  Pdata[10] = acc_fg_rois_;
  Pdata[11] = acc_bg_rois_;
  Pdata[12] = acc_fg_weight_;
  Pdata[13] = acc_bg_weight_;

  return std::make_tuple(RL, RW, P);
}

} // namespace wsl
