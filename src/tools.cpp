#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0)
	{
		cout << "the estimation vector size should not be zero";
		return rmse;
	}

	if (estimations.size() != ground_truth.size())
	{
		cout << "the estimation vector size should equal ground truth vector size";
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i)
	{
		VectorXd term = (estimations[i] - ground_truth[i]);
		term = term.array() * term.array();
		rmse += term;
	}

	rmse /= estimations.size();
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	const float square_root_denominator = sqrt(px * px + py * py);
	if (square_root_denominator == 0)
	{
		cout << "Error: px and py cannot be both zero";
	}
	else
	{
		Hj(0, 0) = px / square_root_denominator;
		Hj(0, 1) = py / square_root_denominator;
		Hj(0, 2) = 0;
		Hj(0, 3) = 0;

		Hj(1, 0) = -py / pow(square_root_denominator, 2);
		Hj(1, 1) = px / pow(square_root_denominator, 2);
		Hj(1, 2) = 0;
		Hj(1, 3) = 0;

		const float nominator_product = vx * py - vy * px;
		Hj(2, 0) = py * nominator_product / pow(square_root_denominator, 3);
		Hj(2, 1) = px * nominator_product / pow(square_root_denominator, 3);
		Hj(2, 2) = Hj(0, 0);
		Hj(2, 3) = Hj(0, 1);
	}

	return Hj;
}
