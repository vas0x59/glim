#include <glim/odometry/odometry_estimation_gpu_gazel.hpp>

extern "C" glim::OdometryEstimationBase* create_odometry_estimation_module() {
  glim::OdometryEstimationGPUGazelParams params;
  return new glim::OdometryEstimationGPUGazel(params);
}