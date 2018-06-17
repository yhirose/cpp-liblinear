#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <mmaplib.h>
#include <peglib.h>
#include <linear.hpp>

using namespace peg;
using namespace std;

template <typename Callback>
bool parse_data(const char* path, Callback callback) {
  parser p(R"(
      ROOT      <-  LINE (nl LINE)* nl?
      LINE      <-  LABEL FEATURES
      FEATURES  <-  FEATURE+
      FEATURE   <-  INDEX ':' VALUE _
      LABEL     <-  < double > _
      INDEX     <-  < integer > _
      VALUE     <-  < double > _
      integer   <-  [0-9]+
      double    <-  [-+]? [0-9]+ ('.' [0-9]+)?
      ~nl       <-  [\r\n]+
      _         <-  ' '*
    )");

  p["LINE"] = [&](const SemanticValues& sv) {
    auto label = sv[0].get<double>();
    auto& features = sv[1].get<vector<tuple<int, double>>>();
    callback(label, features);
  };

  p["FEATURES"] = [](const SemanticValues& sv) {
    return sv.transform<tuple<int, double>>();
  };

  p["FEATURE"] = [](const SemanticValues& sv) {
    return make_tuple(sv[0].get<int>(), sv[1].get<double>());
  };

  p["LABEL"] = [](const SemanticValues& sv) { return stod(sv.token()); };
  p["INDEX"] = [](const SemanticValues& sv) { return stoi(sv.token()); };
  p["VALUE"] = [](const SemanticValues& sv) { return stod(sv.token()); };

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

TEST_CASE("'heart_scale' data test", "[heart_scale]") {
  const auto data_path = "./data/heart_scale";

  linear::problem prob;
  REQUIRE(prepare_problem(data_path, prob));

  linear::parameter param;

  SECTION("find_parameter_C test") {
    auto best_C = linear::find_parameter_C(prob, param);
    REQUIRE(best_C == 1.0);
  }

  SECTION("cross validation test") {
    auto accuracy = cross_validation_accuracy(prob, param);
    REQUIRE(floor(accuracy * 100) == 8407.0);
  }

  SECTION("predict test") {
    linear::model model;
    REQUIRE(model.train(prob, param));

    size_t total = 0;
    size_t correct = 0;

    parse_data(data_path,
               [&](double label, const vector<tuple<int, double>>& features) {
                 total++;
                 linear::feature_nodes nodes;
                 for (const auto& f : features) {
                   nodes.add_feature(get<0>(f), get<1>(f));  // index, value
                 }
                 nodes.end_entry(model.data().bias);
                 auto actual_label = model.predict(nodes.data());
                 if (label == actual_label) {
                   correct++;
                 }
               });

    auto accuracy = (double)correct / (double)total;
    REQUIRE(floor(accuracy * 100) == 84.0);
  }
}
