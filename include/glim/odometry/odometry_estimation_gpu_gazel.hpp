#pragma once

#include <glim/odometry/odometry_estimation_gazel.hpp>

namespace gtsam_points {
class VoxelizedFrame;
class StreamTempBufferRoundRobin;
class CUDAStream;
}  // namespace gtsam_points

namespace glim {

/**
 * @brief Parameters for OdometryEstimationGPU
 */
struct OdometryEstimationGPUGazelParams : public OdometryEstimationGazelParams {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationGPUGazelParams();
  virtual ~OdometryEstimationGPUGazelParams();

  enum class KeyframeUpdateStrategy { OVERLAP, DISPLACEMENT, ENTROPY };

public:
  // Registration params
  double voxel_resolution;
  int voxelmap_levels;
  double voxelmap_scaling_factor;

  int max_num_keyframes;
  int full_connection_window_size;

  // Keyframe management params
  KeyframeUpdateStrategy keyframe_strategy;
  double keyframe_min_overlap;
  double keyframe_max_overlap;
  double keyframe_delta_trans;
  double keyframe_delta_rot;
  double keyframe_entropy_thresh;
};

/**
 * @brief GPU-based tightly coupled LiDAR-IMU odometry
 *
 */
class OdometryEstimationGPUGazel : public OdometryEstimationGazel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationGPUGazel(const OdometryEstimationGPUGazelParams& params = OdometryEstimationGPUGazelParams());
  virtual ~OdometryEstimationGPUGazel() override;

private:
  virtual void create_frame(EstimationFrame::Ptr& frame) override;
  virtual gtsam::NonlinearFactorGraph create_factors(const int current, gtsam::Values& new_values) override;
  virtual void update_frames(const int current, const gtsam::NonlinearFactorGraph& new_factors) override;

  void update_keyframes_overlap(int current);
  void update_keyframes_displacement(int current);
  void update_keyframes_entropy(const gtsam::NonlinearFactorGraph& matching_cost_factors, int current);

private:
  // Keyframe params
  int entropy_num_frames;
  double entropy_running_average;
  std::vector<EstimationFrame::ConstPtr> keyframes;

  // CUDA-related
  std::unique_ptr<gtsam_points::CUDAStream> stream;
  std::unique_ptr<gtsam_points::StreamTempBufferRoundRobin> stream_buffer_roundrobin;
};

}  // namespace glim
