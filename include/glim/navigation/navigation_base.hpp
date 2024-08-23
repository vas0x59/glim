#pragma once


#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>


class NavigationBase {
public:
  NavigationBase();
  virtual ~NavigationBase();

  virtual void insert_imu(double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) = 0;
  virtual void insert_gkv(double stamp, const gtsam::Pose3& pose, const Eigen::Vector3d& v, const Eigen::Vector3d& w) = 0;
  virtual void insert_vd(double stamp, double w, double v) = 0;
  virtual void insert_loc(double stamp, const gtsam::Pose3& pose) = 0;
  virtual void insert_odometry_frame(double stamp, const ) = 0;


};


