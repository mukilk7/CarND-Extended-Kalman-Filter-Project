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
  //Mukil
  if (estimations.size() <= 0 || estimations.size() != ground_truth.size()) {
    return rmse;
  }

  VectorXd rmse(4);
  rmse << 0,0,0,0;

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
  //Mukil
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero
  float px2py2 = px * px + py * py;
  if (px2py2 == 0) {
      cout << "CalculateJacobian () - Error - Division by Zero";
      return Hj;
  }
  float px2py2_sq = pow(px2py2, 0.5);
  float px2py2_15 = pow(px2py2, 1.5);

  //compute the Jacobian matrix
  Hj(0,0) = px / px2py2_sq;
  Hj(0,1) = py / px2py2_sq;
  Hj(1,0) = -py / px2py2;
  Hj(1,1) = px / px2py2;
  Hj(2,0) = (py * (vx * py - vy * px)) / px2py2_15;
  Hj(2,1) = (px * (vy * px - vx * py)) / px2py2_15;
  Hj(2,2) = px / px2py2_sq;
  Hj(2,3) = py / px2py2_sq;

  return Hj;
}
