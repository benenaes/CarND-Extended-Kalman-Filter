#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

class KalmanFilter 
{
public:
	// state vector
	Eigen::VectorXd x_;

	// state covariance matrix
	Eigen::MatrixXd P_;

	// state transition matrix
	Eigen::MatrixXd F_;

	// process covariance matrix
	Eigen::MatrixXd Q_;

	// measurement matrix
	Eigen::MatrixXd H_;

	// measurement covariance matrix
	Eigen::MatrixXd R_;

	/**
	* Constructor
	*/
	KalmanFilter();

	/**
	* Destructor
	*/
	virtual ~KalmanFilter();

	/**
	* Init Initializes Kalman filter
	* @param state_size Size of the state
	*/
	void Init(const std::size_t state_size);

	/**
	* Prediction Predicts the state and the state covariance
	* using the process model
	* @param delta_T Time between k and k+1 in s
	*/
	void Predict();

	/**
	* Updates the state by using standard Kalman Filter equations
	* @param z The measurement at k+1
	*/
	void Update(const Eigen::VectorXd &z);

	/**
	* Updates the state by using Extended Kalman Filter equations
	* @param z The measurement at k+1
	*/
	void UpdateEKF(const Eigen::VectorXd &z);

private:
	/** 
	* Calculate the error in between the prediction and a new measurement in case of a linear measurement function
	* @param z A new measurement
	* @return The error
	*/
	Eigen::VectorXd CalculateLinearError(const Eigen::VectorXd& z);

	/**
	* Calculate the error in between the prediction and a new measurement in case of a non-linear measurement function
	* @param z A new measurement
	* @return The error
	*/
	Eigen::VectorXd CalculateNonLinearError(const Eigen::VectorXd& z);

	/**
	* Update the state x_ and covariance matrix P_ 
	*/
	void UpdateStateAndCovarianceMatrix(const Eigen::VectorXd& ErrorMatrix);

};

#endif /* KALMAN_FILTER_H_ */
