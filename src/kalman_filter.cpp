#include <cmath>
#include <iostream>

#include "kalman_filter.h"

using namespace Eigen;
using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(
	const std::size_t state_size)
{
	x_ = VectorXd(4);
	F_ = MatrixXd(4, 4);
	P_ = MatrixXd(4, 4);

	Q_ = MatrixXd(4, 4);
	Q_ << 0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;
}

void KalmanFilter::Predict() 
{
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

VectorXd KalmanFilter::CalculateLinearError(const VectorXd& z)
{
	const VectorXd z_pred = H_ * x_;
	const VectorXd y = z - z_pred;

	return y;
}

VectorXd KalmanFilter::CalculateNonLinearError(const VectorXd& z)
{
	const double px = x_[0];
	const double py = x_[1];
	const double vx = x_[2];
	const double vy = x_[3];

	double rho = sqrt(pow(px, 2) + pow(py, 2));

	//check division by zero
	if (fabs(rho) < 0.0001)
	{
		cout << "UpdateEKF: rho is close to zero: " << rho << endl;
		rho = 0.0001;
	}

	const double phi = atan2(py, px);
	const double rho_dot = (px * vx + py * vy) / rho;

	VectorXd h(3);
	h << rho, phi, rho_dot;

	const VectorXd y = z - h;

	return y;
}

void KalmanFilter::UpdateStateAndCovarianceMatrix(const Eigen::VectorXd& ErrorMatrix)
{
	const MatrixXd Ht = H_.transpose();
	const MatrixXd S = H_ * P_ * Ht + R_;
	const MatrixXd Si = S.inverse();
	const MatrixXd K = P_ * Ht * Si;

	//new estimate
	x_ = x_ + (K * ErrorMatrix);
	long x_size = x_.size();
	const MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) 
{
	const VectorXd & Error = CalculateLinearError(z);
	UpdateStateAndCovarianceMatrix(Error);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
	const VectorXd & Error = CalculateNonLinearError(z);
	UpdateStateAndCovarianceMatrix(Error);
}
