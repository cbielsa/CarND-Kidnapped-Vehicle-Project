/*
 * particle_filter.cpp
 *
 *  Exercise created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Exercise solved on: May 13, 2017
 *      Author: Carlos Bielsa
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

const double PI = acos(-1.);



void ParticleFilter::init(
	double x, double y, double theta,
	double std[] )
{

	// set the number of particles
	num_particles = 1000;

	// construct normal distributions
	std::default_random_engine gen;
	std::normal_distribution<double> N_x(x, std[0]);
	std::normal_distribution<double> N_y(y, std[1]);
	std::normal_distribution<double> N_theta(theta, std[2]);

	// construct each particle and add to ParticleFilter
	for( int i=0; i<num_particles; ++i )
	{
		// construct particle
		Particle p;
		p.id = i;
		p.x = N_x(gen);
		p.y = N_y(gen);
		p.theta = N_theta(gen);
		p.weight = 1.;

		// push back particle and weight
		particles.push_back(p);
		weights.push_back(p.weight);
	}

	// flag filter as initialized
	is_initialized = true;

	return;
}


void ParticleFilter::prediction(
	double delta_t,
	double std_pos[],
	double velocity, double yaw_rate )
{

	// construct normal noise distributions
	std::default_random_engine gen;
	std::normal_distribution<double> N_x(0., std_pos[0]);
	std::normal_distribution<double> N_y(0., std_pos[1]);
	std::normal_distribution<double> N_theta(0., std_pos[2]);

	// go over particles
	for( int i=0; i<num_particles; ++i )
	{

		// 1. Apply bicycle motion model
		Particle& p = particles[i];

		// case non-negligible yaw rate
		if( fabs(yaw_rate) > 1e-12 )
		{
			double delta_theta = yaw_rate*delta_t;
			double theta_final = p.theta + delta_theta;
			double v_over_yawd = velocity/yaw_rate;
			p.x += v_over_yawd*(sin(theta_final)-sin(p.theta));
			p.y += v_over_yawd*(cos(p.theta)-cos(theta_final));
			p.theta += delta_theta;
		}

		// case negligible yaw rate
		else
		{
			double vdt = velocity*delta_t;
			p.x += vdt*cos(p.theta);
			p.y += vdt*sin(p.theta);
		}

		// 2. Add gaussian noise
		p.x += N_x(gen);
		p.y += N_y(gen);
		p.theta += N_theta(gen);
	}

	return;
}


void ParticleFilter::dataAssociation(
	std::vector<LandmarkObs> predicted,
	std::vector<LandmarkObs>& observations )
{

	// for each observation
	for( int iObs=0; iObs<observations.size(); ++iObs )
	{

		// find closest landmark
		double dMin = std::numeric_limits<double>::max();
		int iClosestLm;

		for( int iLm=0; iLm<predicted.size(); ++iLm )
		{
			// calculate distance between LM prediction and observation
			double d = dist(predicted[iLm].x, predicted[iLm].y, observations[iObs].x, observations[iObs].y);

			if( d<dMin )
			{
				iClosestLm = iLm;
				dMin = d;
			}
		}

		// associate observation to id of closest landmark
		observations[iObs].id = iClosestLm;
	}

	return;
}


void ParticleFilter::updateWeights(
	double sensor_range,
	double std_landmark[], 
	std::vector<LandmarkObs> observations,
	Map map_landmarks )
{

	// for each particle
	for( int iPart=0; iPart<particles.size(); ++iPart )
	{

		// define aliases and auxiliary variables for convenience -----

		// particle pose in global frame
		double& xp = particles[iPart].x;
		double& yp = particles[iPart].y;
		double& thetap = particles[iPart].theta;

		// variables derived from measurement covariance matrix
		if( std_landmark[0]<1e-6 || std_landmark[1]<1e-6 )
		{
			std::cout << "ERROR in ParticleFilter::updateWeights : invalid input std_landmark" << std::endl;
			return;
		}
		double varx = std_landmark[0]*std_landmark[0];
		double vary = std_landmark[1]*std_landmark[1];


		// compute predictions -----

		std::vector<LandmarkObs> predicted;

		// for each landmark
		for( int i=0; i<map_landmarks.landmark_list.size(); ++i )
		{

			// landmark position in global frame
			float& xlm = map_landmarks.landmark_list[i].x_f;
			float& ylm = map_landmarks.landmark_list[i].y_f;

			// calculate relative position
			float p2lmx = xlm-xp;
			float p2lmy = ylm-yp;

			// calculate distance from particle to landmark
			float dist = sqrt(p2lmx*p2lmx + p2lmy*p2lmy);

			// if distance within sensor range, tranform relative vector to vehicle frame,
			// construct observation and add to prediction vector
			if( dist<sensor_range )
			{
				double stheta = sin(thetap);
				double ctheta = cos(thetap);
				LandmarkObs obs;
				obs.x = ctheta*p2lmx +stheta*p2lmy;
				obs.y = -stheta*p2lmx + ctheta*p2lmy;
				predicted.push_back(obs);
			}
		}

		// associate observations to map landmarks
		dataAssociation(predicted, observations);
	
		// calculate weight with multi-variate gaussian probability distribution
		// (there is no need to calculate the denominator, since it is common to all particles)
		double weightp = 1;
		for( int iObs=0; iObs<observations.size(); ++iObs )
		{
			int iPred = observations[iObs].id;

			double dx = observations[iObs].x-predicted[iPred].x;
			double dy = observations[iObs].y-predicted[iPred].y;

			weightp *= exp( -0.5*(dx*dx/varx + dy*dy/vary) );
		}

		// assign unscaled weight to particle in weights vector
		weights[iPart] = weightp;
	}


	// normalize weights vector
	double norm_weights = 0.;
	for( int iPart=0; iPart<num_particles; ++iPart )
		norm_weights += weights[iPart]*weights[iPart];
	norm_weights = sqrt(norm_weights);

	if( norm_weights < 1e-16 )
	{
		std::cout << "ERROR in ParticleFilter::updateWeights : weights vector is almost null" << std::endl;
		std::cout << "norm : " << norm_weights << std::endl;
		exit(1);	
	}

	for( int iPart=0; iPart<num_particles; ++iPart )
	{
		// normalize
		weights[iPart] /= norm_weights;

		// assign weight to particle
		particles[iPart].weight = weights[iPart];
	}

	return;
}


void ParticleFilter::resample()
{

	// construct discrete sampling distribution
	std::default_random_engine gen;
	std::discrete_distribution<int> d(weights.begin(), weights.end());

	// initialize vector of resampled particles
	std::vector<Particle> resampled_particles;

	// populate vector of resampled particles
	for( int i=0; i<num_particles; ++i )
	{
		resampled_particles.push_back( particles[d(gen)] );
	}

	// move to this particles
	particles = std::move(resampled_particles);

	return;
}


void ParticleFilter::write(std::string filename)
{
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
