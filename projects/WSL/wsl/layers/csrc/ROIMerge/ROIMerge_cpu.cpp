#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>
#include <algorithm>
#include <vector>

#include "ROIMerge.h"

namespace wsl {

float getlambda(float iter, float max_iter) {
  float low_bound = 0.01;

  float lambda = (log(iter + low_bound) - log(low_bound)) /
      (log(max_iter + low_bound) - log(low_bound));
  return lambda;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
    return v[i1] > v[i2];
  });

  return idx;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ROIMerge_forward_cpu(
    const at::Tensor& S,
    const at::Tensor& J,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& P) {
  AT_ASSERTM(S.device().is_cpu(), "S must be a CPU tensor");
  AT_ASSERTM(J.device().is_cpu(), "J must be a CPU tensor");
  AT_ASSERTM(C.device().is_cpu(), "C must be a CPU tensor");
  AT_ASSERTM(D.device().is_cpu(), "D must be a CPU tensor");
  AT_ASSERTM(P.device().is_cpu(), "P must be a CPU tensor");

  at::TensorArg S_t{S, "S", 1}, J_t{J, "J", 2}, C_t{C, "C", 3}, D_t{D, "D", 4};

  at::CheckedFrom c = "ROIMerge_forward_cpu";
  at::checkAllSameType(c, {S_t, J_t, C_t, D_t});

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

  AT_ASSERT(P.dim() == 1);
  AT_ASSERT(P.size(0) == 8);
  auto Pdata = P.contiguous().data_ptr<int>();
  int debug_info_ = Pdata[0];
  int display_ = Pdata[1];
  int cur_iter_ = Pdata[2];
  int max_epoch_ = Pdata[3];
  int size_epoch_ = Pdata[4];
  int acc_num_top_id_ = Pdata[5];
  int acc_max_clique_ = Pdata[6];
  int acc_min_clique_ = Pdata[7];

  const int num_roi = C.size(0);
  const int num_class = C.size(1);

  at::Tensor I = at::zeros({num_roi}, C.options().dtype(at::kInt));
  I.fill_(-1);

  auto S_ = S.contiguous(), J_ = J.contiguous(), C_ = C.contiguous(),
       D_ = D.contiguous();

  const float* Sdata = S_.data_ptr<float>();
  const float* Jdata = J_.data_ptr<float>();
  const float* Cdata = C_.data_ptr<float>();
  const float* Ddata = D_.data_ptr<float>();

  auto I_ = I.contiguous();
  int* Idata = I_.data_ptr<int>();

  // sort score
  std::set<int> rois_idx;

  std::vector<float> SSdata;
  SSdata.clear();

  for (int n = 0; n < num_roi; n++) {
    SSdata.push_back(Sdata[n]);
  }
  std::vector<size_t> sort_idx = sort_indexes(SSdata);

  float lambda =
      getlambda(float(cur_iter_) / float(size_epoch_), float(max_epoch_));
  int cur_id = 0;
  int top_k = num_roi > 200 ? 200 : num_roi;

  // merge top
  for (int t = 0; t < top_k; t++) {
    int n = sort_idx[t];
    if (Idata[n] == -1) {
    } else {
      continue;
    }
    Idata[n] = cur_id;

    int end_num = t + 40 > top_k ? top_k : t + 40;

    for (int tt = t; tt < end_num; tt++) {
      int i = sort_idx[tt];
      if (Idata[i] == -1) {
      } else {
        continue;
      }

      bool flag_in_clique = true;

      for (int ttt = t; ttt < end_num; ttt++) {
        int j = sort_idx[ttt];
        if (Idata[j] == cur_id) {
        } else {
          continue;
        }

        if (Jdata[i * num_roi + j] < lambda) {
          flag_in_clique = false;
          break;
        }
      }
      if (flag_in_clique) {
        Idata[i] = cur_id;
      }
    }
    cur_id += 1;
  }

  // for display
  int num_top_id = cur_id;

  // merge rest
  for (int n = 0; n < num_roi; n++) {
    if (Idata[n] == -1) {
    } else {
      continue;
    }

    Idata[n] = cur_id;
    cur_id += 1;
  }

  int num_id = cur_id;

  at::Tensor MC = at::zeros({num_id, num_class}, C.options());
  at::Tensor MD = at::zeros({num_id, num_class}, D.options());
  at::Tensor IC = at::zeros({num_roi}, C.options().dtype(at::kInt));
  IC.fill_(0);

  auto MC_ = MC.contiguous();
  auto MD_ = MD.contiguous();
  auto IC_ = IC.contiguous();

  float* MCdata = MC_.data_ptr<float>();
  float* MDdata = MD_.data_ptr<float>();
  int* ICdata = IC_.data_ptr<int>();

  // count ID
  for (int n = 0; n < num_roi; n++) {
    int id = Idata[n];
    ICdata[id] += 1;
  }

  // for display
  int max_clique = 0;
  int min_clique = top_k;
  for (int i = 0; i < num_top_id; i++) {
    if (ICdata[i] > max_clique) {
      max_clique = ICdata[i];
    }
    if (ICdata[i] < min_clique) {
      min_clique = ICdata[i];
    }
  }
  acc_num_top_id_ += num_top_id;
  acc_max_clique_ += max_clique;
  acc_min_clique_ += min_clique;

  // merge score
  for (int n = 0; n < num_roi; n++) {
    int id = Idata[n];
    for (int c = 0; c < num_class; c++) {
      MCdata[id * num_class + c] += Cdata[n * num_class + c] / ICdata[id];
      MDdata[id * num_class + c] += Ddata[n * num_class + c] / ICdata[id];
    }
  }

  cur_iter_++;

  if (cur_iter_ % display_ == 0) {
    printf(
        "RoIMerge %d\tlambda: %f\tacc_top_num_id: %d\tacc_max_clique: "
        "%d\tacc_min_clique: %d\n",
        cur_iter_,
        lambda,
        acc_num_top_id_ / display_,
        acc_max_clique_ / display_,
        acc_min_clique_ / display_);

    acc_num_top_id_ = 0;
    acc_max_clique_ = 0;
    acc_min_clique_ = 0;
  }

  Pdata[0] = debug_info_;
  Pdata[1] = display_;
  Pdata[2] = cur_iter_;
  Pdata[3] = max_epoch_;
  Pdata[4] = size_epoch_;
  Pdata[5] = acc_num_top_id_;
  Pdata[6] = acc_max_clique_;
  Pdata[7] = acc_min_clique_;

  return std::make_tuple(MC, MD, I, IC, P);
}

std::tuple<at::Tensor, at::Tensor> ROIMerge_backward_cpu(
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& GMC,
    const at::Tensor& GMD,
    const at::Tensor& I,
    const at::Tensor& IC) {
  // Check if input tensors are CPU tensors
  AT_ASSERTM(C.device().is_cpu(), "C must be a CPU tensor");
  AT_ASSERTM(D.device().is_cpu(), "D must be a CPU tensor");
  AT_ASSERTM(GMC.device().is_cpu(), "GMC must be a CPU tensor");
  AT_ASSERTM(GMD.device().is_cpu(), "GMD must be a CPU tensor");
  AT_ASSERTM(I.device().is_cpu(), "I must be a CPU tensor");
  AT_ASSERTM(IC.device().is_cpu(), "IC must be a CPU tensor");

  at::TensorArg C_t{C, "C", 1}, D_t{D, "D", 2}, GMC_t{GMC, "GMC", 2},
      GMD_t{GMD, "GMD", 3}, I_t{I, "I", 4}, IC_t{IC, "IC", 5};

  at::CheckedFrom c = "ROIMerge_backward_cpu";
  at::checkAllSameType(c, {C_t, D_t, GMC_t, GMD_t});
  at::checkAllSameType(c, {I_t, IC_t});

  const int num_roi = C.size(0);
  const int num_class = C.size(1);

  auto GMC_ = GMC.contiguous();
  auto GMD_ = GMD.contiguous();
  auto I_ = I.contiguous();
  auto IC_ = IC.contiguous();

  const float* GMCdata = GMC_.data_ptr<float>();
  const float* GMDdata = GMD_.data_ptr<float>();
  const int* Idata = I_.data_ptr<int>();
  const int* ICdata = IC_.data_ptr<int>();

  at::Tensor GC = at::zeros({num_roi, num_class}, C.options());
  at::Tensor GD = at::zeros({num_roi, num_class}, D.options());

  auto GC_ = GC.contiguous();
  auto GD_ = GD.contiguous();

  float* GCdata = GC_.data_ptr<float>();
  float* GDdata = GD_.data_ptr<float>();

  for (int n = 0; n < num_roi; n++) {
    int id = Idata[n];
    for (int c = 0; c < num_class; c++) {
      GCdata[n * num_class + c] = GMCdata[id * num_class + c] / ICdata[id];
      GDdata[n * num_class + c] = GMDdata[id * num_class + c] / ICdata[id];
    }
  }

  return std::make_tuple(GC, GD);
}

} // namespace wsl
