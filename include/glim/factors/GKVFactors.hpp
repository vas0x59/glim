#pragma once


#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>




namespace glim { namespace factors {
using namespace gtsam;
class GKVShiftedRelativePose3 : public NoiseModelFactorN<Pose3, Pose3> {
  Pose3 pose_;
public:
  using Base = NoiseModelFactorN<Pose3, Pose3>;
public:
  GKVShiftedRelativePose3(Key i, Key j, Pose3 pose, const SharedNoiseModel &model)
          : NoiseModelFactorN<Pose3, Pose3>(model, i, j), pose_(pose) {}
  ~GKVShiftedRelativePose3() override= default;
  Vector evaluateError(const Pose3 &x1, const Pose3 &c,
                       boost::optional<gtsam::Matrix &> H1 = boost::none,
                       boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
    Matrix66 D_shifted_c, D_shifted_x1;

    Pose3 shifted = x1.compose(c, &D_shifted_x1, &D_shifted_c);
    // Matrix66  D_er_x1;
    Matrix66  D_er_shifted;
    Vector6 er = pose_.localCoordinates(shifted, boost::none, &D_er_shifted);

    if (H1) {
      H1->resize(6, 6);
      H1->block<6, 6>(0, 0) = D_er_shifted * D_shifted_x1;
      // H1->block<6, 3>(0, 6).setZero();
    }
    if (H2) {
      H2->resize(6, 6);
      *H2 = D_er_shifted * D_shifted_c;
    }
    return er;
  }
};
}
}