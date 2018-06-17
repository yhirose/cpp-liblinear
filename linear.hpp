//
//  linear.hpp
//
//  Copyright (c) 2018 Yuji Hirose. All rights reserved.
//  MIT License
//

#ifndef _LINEAR_HPP
#define _LINEAR_HPP

#include <iostream>
#include <string>
#include <vector>

namespace linear {

namespace C {
#include <linear.h>
}  // namespace C

namespace detail {

template <typename T>
inline T read(std::istream &is) {
  T data;
  is.read((char *)&data, sizeof(data));
  return data;
}

template <typename T>
inline void write(std::ostream &os, T data) {
  os.write((const char *)&data, sizeof(data));
}

}  // namespace detail

struct parameter : public C::parameter {
  parameter() {
    // default values
    solver_type = C::L2R_L2LOSS_SVC_DUAL;
    C = 1;
    eps = 0.1;
    p = 0.1;
    nr_weight = 0;
    weight_label = nullptr;
    weight = nullptr;
    init_sol = nullptr;
  }

  ~parameter() { destroy_param(this); }

  const char *check_parameter(const C::problem &prob) const {
    return C::check_parameter(&prob, this);
  }
};

struct feature_nodes : public std::vector<C::feature_node> {
  void add_feature(int index, double value) { push_back({index, value}); }

  void end_entry(double bias) {
    if (bias >= 0) {
      push_back({-1, bias});
    }
    push_back({-1, 0});
  }
};

struct problem : public C::problem {
  problem() : max_index_(0) {
    l = 0;
    n = 0;
    bias = -1;
  }

  void begin_entry(double label) {
    l++;
    yv_.push_back(label);
  }

  void add_feature(int index, double value) {
    feature_nodes_.add_feature(index, value);
    if (index > max_index_) {
      max_index_ = index;
    }
  }

  void end_entry() {
    feature_nodes_.end_entry(bias);
    x_space_.emplace_back(std::move(feature_nodes_));
    xv_.push_back(x_space_.back().data());
  }

  void finish() {
    y = yv_.data();
    x = xv_.data();
    if (bias >= 0) {
      auto bias_index = max_index_ + 1;
      n = bias_index;
      for (auto line_index = 0; line_index < l; line_index++) {
        auto &vec = x_space_[line_index];
        vec[vec.size() - 2].index = bias_index;
      }
    } else {
      n = max_index_;
    }
  }

 private:
  int max_index_;
  std::vector<double> yv_;
  std::vector<C::feature_node *> xv_;
  std::vector<feature_nodes> x_space_;
  feature_nodes feature_nodes_;
};

struct model {
  model() : model_(nullptr) {}

  ~model() { free_and_destroy_model(&model_); }

  operator bool() { return model_ != nullptr; }

  const C::model &data() const {
    assert(model_ != nullptr);
    return *model_;
  }

  bool train(const C::problem &prob, const C::parameter &param) {
    model_ = C::train(&prob, &param);
    return model_ != nullptr;
  }

  bool save_model(const char *model_file_name) const {
    assert(model_ != nullptr);
    if (!C::save_model(model_file_name, model_)) {
      return true;
    }
    return false;
  }

  bool load_model(const char *model_file_name) {
    model_ = C::load_model(model_file_name);
    return model_ != nullptr;
  }

  bool save_model_binary(std::ostream &os) const {
    using namespace detail;

    assert(model_ != nullptr);

    const auto &m = *model_;

    write<uint32_t>(os, m.param.solver_type);
    write<uint32_t>(os, m.nr_class);
    if (m.label) {
      for (auto i = 0; i < m.nr_class; i++) {
        write<uint32_t>(os, m.label[i]);
      }
    }
    write<uint32_t>(os, m.nr_feature);
    write<double>(os, m.bias);

    auto w_size = (m.bias >= 0) ? m.nr_feature + 1 : m.nr_feature;
    auto nr_w =
        (m.nr_class == 2 && m.param.solver_type != C::MCSVM_CS) ? 1 : m.nr_class;
    for (auto i = 0; i < w_size; i++) {
      for (auto j = 0; j < nr_w; j++) {
        write<double>(os, m.w[i * nr_w + j]);
      }
    }

    return true;
  }

  bool load_model_binary(std::istream &is) {
    using namespace detail;

    auto m = (C::model *)malloc(sizeof(C::model));

    m->param.solver_type = read<uint32_t>(is);
    m->nr_class = read<uint32_t>(is);
    m->label =
        m->nr_class > 0 ? (int *)malloc(sizeof(int) * m->nr_class) : nullptr;
    for (auto i = 0; i < m->nr_class; i++) {
      m->label[i] = read<uint32_t>(is);
    }
    m->nr_feature = read<uint32_t>(is);
    m->bias = read<double>(is);

    auto w_size = (m->bias >= 0) ? m->nr_feature + 1 : m->nr_feature;
    auto nr_w = (m->nr_class == 2 && m->param.solver_type != C::MCSVM_CS)
                    ? 1
                    : m->nr_class;
    m->w = (double *)malloc(sizeof(double) * w_size * nr_w);

    for (auto i = 0; i < w_size; i++) {
      for (auto j = 0; j < nr_w; j++) {
        m->w[i * nr_w + j] = read<double>(is);
      }
    }

    model_ = m;

    return model_ != nullptr;
  }

  double predict(const C::feature_node *x) const {
    std::vector<double> dummy(model_->nr_class, 0.0);
    return C::predict_values(model_, x, dummy.data());
  }

  double predict_probability(const C::feature_node *x,
                             std::vector<double> &prob_estimates) const {
    prob_estimates.resize(model_->nr_class);
    return C::predict_probability(model_, x, prob_estimates.data());
  }

  int get_nr_feature() const { return C::get_nr_feature(model_); }

  void get_labels(std::vector<int> &labels) const {
    assert(model_ != nullptr);
    auto nr_class = C::get_nr_class(model_);
    labels.resize(nr_class);
    C::get_labels(model_, labels.data());
  }

  int find_label_index(int label) const {
    if (model_->label != NULL) {
      for (int i = 0; i < model_->nr_class; i++) {
        if (model_->label[i] == label) {
          return i;
        }
      }
    }
    return -1;
  }

 private:
  C::model *model_;
};

inline double find_parameter_C(const C::problem &prob, const C::parameter &param,
                               int nr_fold = 5, double start_C = -1.0) {
  double best_C, best_rate;
  C::find_parameter_C(&prob, &param, nr_fold, start_C, 1024, &best_C, &best_rate);
  return best_C;
}

inline double cross_validation_accuracy(const C::problem &prob,
                                        const C::parameter &param,
                                        int nr_fold = 5) {
  std::vector<double> target(prob.l, 0.0);
  C::cross_validation(&prob, &param, nr_fold, target.data());

  if (param.solver_type == C::L2R_L2LOSS_SVR ||
      param.solver_type == C::L2R_L1LOSS_SVR_DUAL ||
      param.solver_type == C::L2R_L2LOSS_SVR_DUAL) {
    // skip regression_model...
  } else {
    int total_correct = 0;
    for (int i = 0; i < prob.l; i++) {
      if (target[i] == prob.y[i]) {
        ++total_correct;
      }
    }
    return 100.0 * total_correct / prob.l;
  }

  return -1.0;
}

}  // namespace linear

#endif  // _LINEAR_S_HPP
