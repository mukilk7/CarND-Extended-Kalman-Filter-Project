#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::GenericUpdate(const VectorXd &z, const VectorXd &z_pred) {
  /*
  * Generic kalman filter update function. The z_pred
  * needs to be custom calculated for each filter type.
  */
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd CartesianToPolar(const VectorXd &z) {
  /*
  * Converts Cartesian data to polar coordinates.
  */
  VectorXd polar = VectorXd(3);
  float ro = sqrt((z[0] * z[0] + z[1] * z[1]));
  float phi = 0.0f;
  if (z[0] != 0) {
    //Note: skipping angle normalization as atan2
    //automatically  gives output in (-pi, pi) range.
    phi = atan2(z[1], z[0]);
  }
  float rodot = 0.0f;
  if (ro != 0) {
    rodot = (z[0] * z[2] + z[1] * z[3]) / ro;
  }
  polar << ro, phi, rodot;
  return polar;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  GenericUpdate(z, z_pred);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  VectorXd z_pred = CartesianToPolar(x_);
  GenericUpdate(z, z_pred);
}
