//
//  linear.hpp
//
//  Copyright (c) 2022 Yuji Hirose. All rights reserved.
//  MIT License
//

#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace linear {

namespace C {
#include <linear.h>
} // namespace C

namespace detail {

template <typename T> inline T read(std::istream &is) {
  T data;
  is.read((char *)&data, sizeof(data));
  return data;
}

template <typename T> inline void write(std::ostream &os, T data) {
  os.write((const char *)&data, sizeof(data));
}

} // namespace detail

struct parameter : public C::parameter {
  parameter() {
    // default values
    solver_type = C::L2R_L2LOSS_SVC_DUAL;
    C = 1;
    p = 0.1;
    nu = 0.5;
    eps = 0.1;
    nr_weight = 0;
    regularize_bias = 1;
    weight_label = NULL;
    weight = NULL;
    init_sol = NULL;
  }

  ~parameter() { destroy_param(this); }
};

struct feature_nodes : public std::vector<C::feature_node> {
  void add_feature(int index, double value) { push_back({index, value}); }

  void end_entry(const C::model &model) {
    if (model.bias >= 0) { push_back({model.nr_feature + 1, model.bias}); }
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
    feature_nodes_.push_back({index, value});
    if (index > max_index_) { max_index_ = index; }
  }

  void end_entry() {
    if (bias >= 0) {
      feature_nodes_.push_back({-1, bias}); // `index` will be set in `finish()`
    }
    feature_nodes_.push_back({-1, 0});

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
        auto &feature_nodes = x_space_[line_index];
        feature_nodes[feature_nodes.size() - 2].index = bias_index;
      }
    } else {
      n = max_index_;
    }
  }

private:
  int max_index_;
  std::vector<double> yv_;
  std::vector<C::feature_node *> xv_;
  std::vector<std::vector<C::feature_node>> x_space_;
  std::vector<C::feature_node> feature_nodes_;
};

struct model {
  model() : model_(nullptr) {}

  ~model() { free_and_destroy_model(&model_); }

  operator bool() { return model_ != nullptr; }

  const C::model &data() const {
    assert(model_ != nullptr);
    return *model_;
  }

  C::model &data() {
    assert(model_ != nullptr);
    return *model_;
  }

  bool train(const C::problem &prob, const C::parameter &param) {
    if (C::check_parameter(&prob, &param) != NULL) { return false; }
    C::set_print_string_function(&print_null);
    model_ = C::train(&prob, &param);
    return model_ != nullptr;
  }

  bool save_model(const char *model_file_name) const {
    assert(model_ != nullptr);
    if (!C::save_model(model_file_name, model_)) { return true; }
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
    auto nr_w = (m.nr_class == 2 && m.param.solver_type != C::MCSVM_CS)
                    ? 1
                    : m.nr_class;
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

  double predict_values(const C::feature_node *x,
                        std::vector<double> &dec_values) const {
    dec_values.resize(model_->nr_class);
    return C::predict_values(model_, x, dec_values.data());
  }

  double predict_probability(const C::feature_node *x,
                             std::vector<double> &prob_estimates) const {
    prob_estimates.resize(model_->nr_class);
    return C::predict_probability(model_, x, prob_estimates.data());
  }

  int get_nr_feature() const { return C::get_nr_feature(model_); }

  int get_nr_class() const { return C::get_nr_class(model_); }

  const C::parameter &get_param() const { return model_->param; }

  double get_bias() const { return model_->bias; }

  void get_labels(std::vector<int> &labels) const {
    assert(model_ != nullptr);
    auto nr_class = C::get_nr_class(model_);
    labels.resize(nr_class);
    C::get_labels(model_, labels.data());
  }

  int find_label_index(int label) const {
    if (model_->label != NULL) {
      for (int i = 0; i < model_->nr_class; i++) {
        if (model_->label[i] == label) { return i; }
      }
    }
    return -1;
  }

private:
  inline static void print_null(const char * /*s*/) {}
  C::model *model_;
};

} // namespace linear
