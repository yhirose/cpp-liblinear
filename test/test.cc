#define CATCH_CONFIG_MAIN
#include <mmaplib.h>
#include <peglib.h>

#include <catch.hpp>
#include <linear.hpp>

using namespace peg;
using namespace std;
using namespace linear;

double cross_validation_accuracy(const C::problem& prob,
                                 const C::parameter& param, int nr_fold = 5) {
  if (C::check_parameter(&prob, &param) != NULL) {
    return -1.0;
  }

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

template <typename Callback>
bool parse_data(const char* path, Callback callback) {
  parser p(R"(
      ROOT      <-  LINE (nl LINE)* nl?
      LINE      <-  LABEL FEATURES
      FEATURES  <-  FEATURE+

      FEATURE   <-  INDEX ':' VALUE _
      LABEL     <-  double _
      INDEX     <-  integer _
      VALUE     <-  double _

      integer   <-  [0-9]+
      double    <-  [-+]? [0-9]+ ('.' [0-9]+ ('e-' [0-9]+)?)?
      ~nl       <-  [\r\n]+
      _         <-  ' '*
    )");

  p["LINE"] = [&](const SemanticValues& sv) {
    auto label = any_cast<double>(sv[0]);
    const auto& features = any_cast<vector<tuple<int, double>>>(sv[1]);
    callback(label, features);
  };

  p["FEATURES"] = [](const SemanticValues& sv) {
    return sv.transform<tuple<int, double>>();
  };

  p["FEATURE"] = [](const SemanticValues& sv) {
    return make_tuple(any_cast<int>(sv[0]), any_cast<double>(sv[1]));
  };

  p["integer"] = [](const SemanticValues& sv) { return stoi(sv.token()); };
  p["double"] = [](const SemanticValues& sv) { return stod(sv.token()); };

  mmaplib::MemoryMappedFile mmap(path);
  return p.parse_n(mmap.data(), mmap.size());
}

bool prepare_problem(const char* path, linear::problem& prob) {
  auto ret = parse_data(
      path, [&](double label, const vector<tuple<int, double>>& features) {
        prob.begin_entry(label);
        for (const auto& f : features) {
          prob.add_feature(get<0>(f), get<1>(f));  // index, value
        }
        prob.end_entry();
      });

  if (ret) {
    prob.finish();
  }

  return ret;
}

double get_accuracy(const linear::model& model, const char* data_path) {
  size_t total = 0;
  size_t correct = 0;

  parse_data(data_path,
             [&](double label, const vector<tuple<int, double>>& features) {
               total++;
               linear::feature_nodes nodes;
               for (const auto& f : features) {
                 nodes.add_feature(get<0>(f), get<1>(f));  // index, value
               }
               nodes.end_entry(model.data());
               auto actual_label = model.predict(nodes.data());
               if (label == actual_label) {
                 correct++;
               }
             });

  auto accuracy = (double)correct / (double)total;

  return accuracy;
}

TEST_CASE("'heart' dataset test", "[heart]") {
  const auto data_path = "../data/heart.scale";

  linear::problem prob;
  REQUIRE(prepare_problem(data_path, prob));

  linear::parameter param;

  SECTION("cross validation test") {
    auto accuracy = cross_validation_accuracy(prob, param);
    CHECK(floor(accuracy * 100) == 8370.0);
  }

  SECTION("predict test") {
    linear::model model;
    REQUIRE(model.train(prob, param));

    auto accuracy = get_accuracy(model, data_path);
    CHECK(floor(accuracy * 100) == 84.0);
  }
}

TEST_CASE("'iris' dataset test", "[iris]") {
  const auto data_path = "../data/iris.scale";

  linear::problem prob;
  prob.bias = 0.9;
  REQUIRE(prepare_problem(data_path, prob));

  linear::parameter param;
  param.C = 8.0;

  SECTION("cross validation test") {
    auto accuracy = cross_validation_accuracy(prob, param);
    CHECK(floor(accuracy * 100) == 9466.0);
  }

  SECTION("predict test") {
    linear::model model;
    REQUIRE(model.train(prob, param));

    auto accuracy = get_accuracy(model, data_path);
    CHECK(floor(accuracy * 100) == 96.0);
  }
}

