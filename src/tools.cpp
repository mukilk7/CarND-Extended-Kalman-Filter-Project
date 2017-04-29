#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() <= 0 || estimations.size() != ground_truth.size()) {
    return rmse;
  }

  //Compute sum of squared residuals
  for (int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse = rmse + residual;
  }
  //Average the sum
  rmse = rmse / estimations.size();
  //Compute square root of average sum
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero
  float px2py2 = px * px + py * py;
  if (px2py2 < 0.0001) {
      std::cout << "CalculateJacobian () - Error - Division by Zero";
      return Hj;
  }
  float px2py2_sq = sqrt(px2py2);
  float px2py2_15 = px2py2 * px2py2_sq;

  //compute the Jacobian matrix
  Hj(0,0) = px / px2py2_sq;
  Hj(0,1) = py / px2py2_sq;
  Hj(0,2) = 0;
  Hj(0,3) = 0;
  Hj(1,0) = -py / px2py2;
  Hj(1,1) = px / px2py2;
  Hj(1,2) = 0;
  Hj(1,3) = 0;
  Hj(2,0) = (py * (vx * py - vy * px)) / px2py2_15;
  Hj(2,1) = (px * (vy * px - vx * py)) / px2py2_15;
  Hj(2,2) = px / px2py2_sq;
  Hj(2,3) = py / px2py2_sq;

  return Hj;
}
