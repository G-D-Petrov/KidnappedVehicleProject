/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	// define normal distributions for sensor noise
	normal_distribution<double> distribution_x(0, std[0]);
	normal_distribution<double> distribution_y(0, std[1]);
	normal_distribution<double> distribution_theta(0, std[2]);

	// init particles
	for (int i = 0; i < num_particles; i++) 
	{
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;

		// add noise
		p.x += distribution_x(gen);
		p.y += distribution_y(gen);
		p.theta += distribution_theta(gen);

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define normal distributions for sensor noise
	normal_distribution<double> distribution_x(0, std_pos[0]);
	normal_distribution<double> distribution_y(0, std_pos[1]);
	normal_distribution<double> distribution_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++)
	{

		// calculate new state
		if (fabs(yaw_rate) < 0.00001)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add noise
		particles[i].x += distribution_x(gen);
		particles[i].y += distribution_y(gen);
		particles[i].theta += distribution_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++) 
	{
		LandmarkObs o = observations[i];

		double min_dist = numeric_limits<double>::max();

		int map_id = -1;

		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs p = predicted[j];
			double cur_dist = dist(o.x, o.y, p.x, p.y);

			if (cur_dist < min_dist)
			{
				min_dist = cur_dist;
				map_id = p.id;
			}
		}

		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
	const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++)
	{
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		vector<LandmarkObs> predictions;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) 
		{
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			if (fabs(landmark_x - particle_x) <= sensor_range && fabs(landmark_y - particle_y) <= sensor_range) 
			{
				predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		vector<LandmarkObs> transformed_os;
		for (unsigned int j = 0; j < observations.size(); j++) 
		{
			double t_x = cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y + particle_x;
			double t_y = sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y + particle_y;
			transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		dataAssociation(predictions, transformed_os);

		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < transformed_os.size(); j++) 
		{
			double observation_x = transformed_os[j].x;
			double observation_y = transformed_os[j].y;
			double predicted_x = 0.0;
			double predicted_y = 0.0;
			int associated_prediction = transformed_os[j].id;
			
			for (unsigned int k = 0; k < predictions.size(); k++)
			{
				if (predictions[k].id == associated_prediction) 
				{
					predicted_x = predictions[k].x;
					predicted_y = predictions[k].y;
				}
			}

			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = (1 / (2 * M_PI*s_x*s_y)) * exp(-(pow(predicted_x - observation_x, 2) / (2 * pow(s_x, 2)) + (pow(predicted_y - observation_y, 2) / (2 * pow(s_y, 2)))));

			particles[i].weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;
	vector<double> new_weights;

	for (int i = 0; i < num_particles; i++) 
	{
		new_weights.push_back(particles[i].weight);
	}

	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);

	double max_weight = *max_element(new_weights.begin(), new_weights.end());

	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	for (int i = 0; i < num_particles; i++) 
	{
		beta += unirealdist(gen) * 2.0;

		while (beta > new_weights[index])
		{
			beta -= new_weights[index];
			index = (index + 1) % num_particles;
		}

		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
