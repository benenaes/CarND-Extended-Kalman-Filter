#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() 
{
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	H_laser_ <<
		1, 0, 0, 0,
		0, 1, 0, 0;

	Hj_ = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
		0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
		0, 0.0009, 0,
		0, 0, 0.09;

	ekf_.Init(4);

	//the initial transition matrix F_
	ekf_.F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;

	// the initial state covariance matrix P_
	ekf_.P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) 
	{
		// first measurement

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
		{
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			const double px = cos(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[0];
			const double py = sin(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[0];
			// For initialization, I assume the velocity of the detected object is along the axis 
			// defined by phi (measurement_pack.raw_measurements_[1])
			// This resulted in better performance than initializing with zeroes on dataset 2 (where the 1st measurement is RADAR).
			const double vx = cos(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[2];
			const double vy = sin(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[2];
			ekf_.x_ << px, py, vx, vy;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
		{
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		}

		previous_timestamp_ = measurement_pack.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		cout << "Initialized !" << endl;
		return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	static const double noise_ax = 9;
	static const double noise_ay = 9;

	// dt - expressed in seconds
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	
	previous_timestamp_ = measurement_pack.timestamp_;

	// Update the state transition matrix F according to the new elapsed time.
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	// Update the process noise covariance matrix.
	cout << "dt: " << dt << endl;
	const float dt_squared = dt * dt;
	const float dt_pow4 = dt_squared * dt_squared;
	cout << "dt_pow4: " << dt_pow4 << endl;
	ekf_.Q_(0, 0) = dt_pow4 * 0.25 * noise_ax;
	ekf_.Q_(1, 1) = dt_pow4 * 0.25 * noise_ay;
	ekf_.Q_(0, 2) = dt_squared * dt * 0.5 * noise_ax;
	ekf_.Q_(1, 3) = dt_squared * dt * 0.5 * noise_ay;
	ekf_.Q_(2, 0) = dt_squared * dt * 0.5 * noise_ax;
	ekf_.Q_(3, 1) = dt_squared * dt * 0.5 * noise_ay;
	ekf_.Q_(2, 2) = dt_squared * noise_ax;
	ekf_.Q_(3, 3) = dt_squared * noise_ay;

	ekf_.Predict();

	/*****************************************************************************
	*  Update
	****************************************************************************/

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
	{
		// Radar updates
		ekf_.R_ = R_radar_;
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	} 
	else 
	{
		// Laser updates
		ekf_.R_ = R_laser_;
		ekf_.H_ = H_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}

	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
