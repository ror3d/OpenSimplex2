#pragma once

#include "../opensimplex2s.hpp"

#include <iostream>
#include <chrono>

using namespace osn;

#define N_VALUES (1024*1024)

#define ITERATIONS 100


float values[N_VALUES];


void main()
{
	OpenSimplex2S<2, osn::Mode::Standard_2D> osn2d;

	float ph2 = 1.32471795724474602596;

	std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

	for (size_t iter = 0; iter < ITERATIONS; ++iter)
		for (size_t i = 0; i < N_VALUES; ++i)
		{
			float x = (float(i) / ph2);
			x = x - std::floor(x);
			float y = (float(i) / (ph2 * ph2));
			y = y - std::floor(y);
			values[i] = osn2d(x * 30 - 15, y * 50 - 25);
		}

	std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();

	std::cout << "2D OSN for " << N_VALUES << " values took " << std::chrono::duration<float>(end - start).count()/ITERATIONS << " seconds\n";


	//////////////////////////////////////////////////

	OpenSimplex2S<3, osn::Mode::Classic_3D> osn3d;

	float ph3 = 1.22074408460575947536;

	start = std::chrono::high_resolution_clock::now();

	for (size_t iter = 0; iter < ITERATIONS; ++iter)
		for (size_t i = 0; i < N_VALUES; ++i)
		{
			float x = (float(i) / ph3);
			x = x - std::floor(x);
			float y = (float(i) / (ph3 * ph3));
			y = y - std::floor(y);
			float z = (float(i) / (ph3 * ph3 * ph3));
			z = z - std::floor(z);
			values[i] = osn3d(x * 30 - 15, y * 50 - 25, z * 42 - 21);
		}

	end = std::chrono::high_resolution_clock::now();

	std::cout << "3D OSN for " << N_VALUES << " values took " << std::chrono::duration<float>(end - start).count()/ITERATIONS << " seconds\n";


	//////////////////////////////////////////////////

	OpenSimplex2S<4, osn::Mode::Classic_4D> osn4d;

	float ph4 = 1.167303978412138;

	start = std::chrono::high_resolution_clock::now();

	for (size_t iter = 0; iter < ITERATIONS; ++iter)
		for (size_t i = 0; i < N_VALUES; ++i)
		{
			float x = (float(i) / ph4);
			x = x - std::floor(x);
			float y = (float(i) / (ph4 * ph4));
			y = y - std::floor(y);
			float z = (float(i) / (ph4 * ph4 * ph4));
			z = z - std::floor(z);
			float w = (float(i) / (ph4 * ph4 * ph4 * ph4));
			w = w - std::floor(w);
			values[i] = osn4d(x * 30 - 15, y * 50 - 25, z * 42 - 21, w * 23 - 12);
		}

	end = std::chrono::high_resolution_clock::now();

	std::cout << "3D OSN for " << N_VALUES << " values took " << std::chrono::duration<float>(end - start).count()/ITERATIONS << " seconds\n";
}

