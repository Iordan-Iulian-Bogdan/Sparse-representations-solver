#pragma once
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <array>
#include <immintrin.h>
#include <CL/cl.hpp>
#include "AVX_functions.h"

using namespace std::chrono;
using namespace std;

class sparse
{
public:

	uint32_t n = 0;
	uint32_t m = 0;
	uint32_t T = omp_get_num_procs();
	uint32_t old_n = 0;
	uint32_t old_m = 0;
	float max_eig = 0.0f;
	float error = 0.0f;
	vector<vector<float>> A;
	vector<vector<float>> A_t;
	vector<float> b;
	vector<float> x;
	vector<float> sol;
	double solve_time = 0.0f;
	bool is_max_eig_set = false;

	sparse(string A_name, string b_name)
	{
		//	opneing the files
		std::ifstream fin1(A_name, std::ios::binary);
		std::ifstream fin2(b_name, std::ios::binary);

		if (!fin1.is_open())
		{
			throw std::invalid_argument("Error opening the file");
		}

		fin1.read(reinterpret_cast<char*>(&n), sizeof(n));
		fin1.read(reinterpret_cast<char*>(&m), sizeof(m));

		//	redefining the sizes such that they are multiples of 32 for performance reasons
		//	this is done in order to processes the data into chucks of 8(cpu) or 32(gpu) 
		old_n = n;

		if (n % 32 != 0)
		{
			n = n + (32 - n % 32);
		}

		old_m = m;

		if (m % 32 != 0)
		{
			m = m + (32 - m % 32);
		}

		//	resizing to the new correct sizes
		this->A.resize(m, vector<float>(n));
		this->A_t.resize(n, vector<float>(m));
		this->b.resize(m);
		this->x.resize(n);

		//	filling A and b with zeros before reading into them the actual data
		//	the extra lines and collums will not affect the computation since they are now filled with zeros

		fill(b.begin(), b.end(), 0.0f);

		for (unsigned int i = 0; i < A.size(); ++i)
			fill(A[i].begin(), A[i].end(), 0.0f);

		//	reading the data from disk
		for (unsigned int i = 0; i < A.size(); ++i)
			fin1.read(reinterpret_cast<char*>(A[i].data()), (A[i].size() - (n - old_n)) * sizeof(float));
		fin1.close();

		fin2.read(reinterpret_cast<char*>(b.data()), (b.size()) * sizeof(float));
		fin2.close();

		mat_transpose(A, A_t, T);
	}

	//	sets the number of threads that the CPU solver can use 
	//	by default this is set to the maximum number of threads available on the machine
	void set_number_of_threads(int T)
	{
		this->T = T;
	}

	vector<vector<float>> get_dictionary()
	{
		return this->A;
	}

	vector<float> get_measurement()
	{
		return this->b;
	}

	void set_max_eig(float max_eig)
	{
		this->max_eig = max_eig;
		is_max_eig_set = true;
	}

protected:

	//	function which aproximates the maximum singular value using the power method 
	//	(required for the convergence of all three algorithms)
	//	is GPU accelarated (namely the matrix multiplication which is very expensive for large matrices)
	float power_method_gpu()
	{
		//	finding an avalaible GPU device and setting up OpenCL
		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		cl::Platform default_platform = all_platforms[0];
		std::vector<cl::Device> all_devices;
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
		cl::Device default_device = all_devices[0];
		cl::Context context(default_device);
		std::ifstream src("gpu_kernels.cl");//this is from where the kernels are read
		std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
		cl::Program::Sources sources;
		sources.push_back({ str.c_str(),str.length() });
		cl::Program program(context, sources);
		program.build({ default_device });

		vector<float> flat_A;
		vector<float> flat_A_t;
		vector<vector<float>> X(m, vector<float>(m));
		vector<float> flat_X;

		//	"flattening" the arrays for the OpenCL kernels since they take unidimensional data as input
		for (unsigned int i = 0; i < A_t.size(); ++i)
			for (unsigned int j = 0; j < A_t[0].size(); ++j)
			{
				flat_A.push_back(A_t[i][j]);
			}

		for (unsigned int i = 0; i < A.size(); ++i)
			for (unsigned int j = 0; j < A[0].size(); ++j)
			{
				flat_A_t.push_back(A[i][j]);
			}

		for (unsigned int i = 0; i < X.size(); ++i)
			for (unsigned int j = 0; j < X[0].size(); ++j)
			{
				flat_X.push_back(X[i][j]);
			}

		//	defining the buffer which will exist on the GPU memory
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A_t.size());
		cl::Buffer buffer_X(context, CL_MEM_READ_WRITE, sizeof(float) * flat_X.size());

		cl::CommandQueue queue(context, default_device);

		//	compiling the kernel
		cl::Kernel kernel_mat_mat_mul_gpu = cl::Kernel(program, "mat_mat_mul_gpu");

		//	writting tot the buffers
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(float) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_X, CL_TRUE, 0, sizeof(float) * flat_X.size(), flat_X.data());

		//	setting the kernel arguments
		kernel_mat_mat_mul_gpu.setArg(0, m);
		kernel_mat_mat_mul_gpu.setArg(1, n);
		kernel_mat_mat_mul_gpu.setArg(2, buffer_A);
		kernel_mat_mat_mul_gpu.setArg(3, buffer_A_t);
		kernel_mat_mat_mul_gpu.setArg(4, buffer_X);

		//	we'll use multiples of 32 for the sizes of the workgroups 
		//	this matches the characteristics of a lot of the hardware avaialble
		const int TS = 32;

		//launching the kernel in execution
		queue.enqueueNDRangeKernel(kernel_mat_mat_mul_gpu, cl::NullRange, cl::NDRange(m, m), cl::NDRange(32, 32));

		//	reading back the result
		queue.enqueueReadBuffer(buffer_X, CL_TRUE, 0, sizeof(float) * flat_X.size(), flat_X.data());
		queue.finish();

		uint64_t k = 0;

		//	"unflattening" the data back
		for (unsigned int i = 0; i < X.size(); ++i)
			for (unsigned int j = 0; j < X[0].size(); ++j)
			{
				X[i][j] = flat_X[k++];
			}

		vector<float> b_k(m);
		vector<float> aux(m);
		vector<float> b_k1(m);
		float norm_b_k1 = 0.0f;
		float eig = 0.0f;

		fill(b_k.begin(), b_k.end(), 1.0f);

		//	bulk of the algorithm
		for (unsigned int i = 0; i < 10; i++)
		{
			mat_vec_mul_avx(X, b_k, this->T);
			b_k1 = b_k;
			norm_b_k1 = norm(b_k1);
			aux = b_k1;
			vec_scalar_avx(b_k1, (1 / norm_b_k1), this->T);
			b_k = b_k1;
			b_k1 = aux;
		}

		aux = b_k;
		mat_vec_mul_avx(X, b_k, this->T);
		b_k = vec_mul_avx(b_k, aux);
		eig = reduction_avx(b_k);
		aux = vec_mul_avx(aux, aux);
		eig = eig / reduction_avx(aux);

		return eig;
	}


	//	same as above except is not GPU accelarated 
	float power_method_cpu()
	{
		vector<vector<float>> X(m, vector<float>(m));
		vector<float> b_k(m);
		vector<float> aux(m);
		vector<float> b_k1(m);
		float norm_b_k1 = 0.0f;
		float eig = 0.0f;
		mat_mat_mul_avx(A, A, X, this->T);

		fill(b_k.begin(), b_k.end(), 1.0f);

		for (unsigned int i = 0; i < 10; i++)
		{
			mat_vec_mul_avx(X, b_k, this->T);
			b_k1 = b_k;
			norm_b_k1 = norm(b_k1);
			aux = b_k1;
			vec_scalar_avx(b_k1, (1 / norm_b_k1), this->T);
			b_k = b_k1;
			b_k1 = aux;
		}

		aux = b_k;
		mat_vec_mul_avx(X, b_k, this->T);
		b_k = vec_mul_avx(b_k, aux);
		eig = reduction_avx(b_k);
		aux = vec_mul_avx(aux, aux);
		eig = eig / reduction_avx(aux);

		return eig;
	}

	virtual void solve_cpu() {}

	virtual void solve_gpu() {}
};

class adm : public sparse
{
public:

	int iterations = 0;
	bool is_solved = false;
	float gamma = 0.0f;
	float beta = 0.0f;
	float tau = 0.0f;

	adm(string A_name, string b_name, float beta, float tau, int iterations) :sparse(A_name, b_name)
	{
		//	tau and beta should tipically be small i.e 0.001
		this->iterations = iterations;
		this->beta = beta;
		this->tau = tau;
	}

	void solve_cpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		//	checking if the maximum singular value has been set otherwise we'll compute it
		if (!is_max_eig_set)
			max_eig = power_method_cpu();

		//	gamma is required to be less than 2 in order to ensure convergence
		//	note that if the maximum singular value are too big this can no longer be true
		//	and you'd need to change tau and beta
		this->gamma = 1.99f - (this->tau * max_eig);

		vector<float>aux1(n), aux2(n), aux3(n);
		vector<float>y(m);
		vector<float>r(m);
		mat_scalar_avx(A_t, tau, this->T);

		//	Implemeting the following ADM algorithm :
		//	Input: ?, ?, ? dictionary A, measurement b, x = 0, y = 0
		//	While not converge
		//	x(k)?shrink(x(k) - ?A * (Ax(k) - b - y(k) / ?), ? / ?)
		//	y(k + 1)?y(k) - ??(Ax(k + 1) - b)
		//	end while
		//	Output: x(k)

		for (int i = 0; i <= iterations; i++)
		{
			aux1 = x;
			aux2 = y;

			mat_vec_mul_avx(A, aux1, this->T);
			vec_sub_avx(aux1, b, this->T);
			vec_scalar_avx(aux2, (1 / beta), this->T);
			vec_sub_avx(aux1, aux2, this->T);
			mat_vec_mul_avx(A_t, aux1, this->T);
			vec_sub_avx(x, aux1, this->T);
			shrink(x, (tau / beta), this->T);

			aux1 = x;
			mat_vec_mul_avx(A, aux1, this->T);
			vec_sub_avx(aux1, b, this->T);
			vec_scalar_avx(aux1, (gamma * beta), this->T);
			vec_sub_avx(y, aux1, this->T);
		}

		//	the solution is cut down to the correct size, remember that we resized A and b when we read them
		vector<float>aux(x.begin(), x.begin() + old_n);
		sol = aux;
		is_solved = true;
		vector<float> err = x;
		mat_vec_mul_avx(A, err, this->T);
		vec_sub_avx(err, b, this->T);
		this->error = norm(err); //	computing the error A * x - b
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	//	same as above except all the computations inside the main loop are now replaced 
	//	with the equivalent OpenCL kernels  
	void solve_gpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!is_max_eig_set)
			max_eig = power_method_gpu();

		this->gamma = 1.99f - (this->tau * max_eig);

		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		cl::Platform default_platform = all_platforms[0];
		std::vector<cl::Device> all_devices;
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
		cl::Device default_device = all_devices[0];
		cl::Context context(default_device);
		std::ifstream src("gpu_kernels.cl");
		std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
		cl::Program::Sources sources;
		sources.push_back({ str.c_str(),str.length() });
		cl::Program program(context, sources);
		program.build({ default_device });

		vector<float> flat_A;
		vector<float> flat_A_t;
		vector<float>y(m);
		int t = 32;
		int m1 = 32;

		for (unsigned int i = 0; i < A_t.size(); ++i)
			for (unsigned int j = 0; j < A_t[0].size(); ++j)
			{
				flat_A.push_back(A_t[i][j]);
			}

		for (unsigned int i = 0; i < A.size(); ++i)
			for (unsigned int j = 0; j < A[0].size(); ++j)
			{
				flat_A_t.push_back(A[i][j]);
			}

		vec_scalar_avx(flat_A_t, tau, this->T);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A.size());
		cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_y(context, CL_MEM_READ_WRITE, sizeof(float) * y.size());
		cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res2(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A_t.size());
		cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(float) * b.size());

		cl::CommandQueue queue(context, default_device);

		cl::Kernel kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu");
		cl::Kernel kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu");
		cl::Kernel kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu");
		cl::Kernel kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu");

		queue.enqueueWriteBuffer(buffer_y, CL_TRUE, 0, sizeof(float) * y.size(), y.data());
		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(float) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(float) * b.size(), b.data());
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_y, CL_TRUE, 0, sizeof(float) * y.size(), y.data());
		queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());

		for (int i = 0; i <= iterations; i++)
		{
			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_x);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, m);
			kernel_mat_vec_mul_gpu.setArg(5, n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res);
			kernel_vec_sub_gpu.setArg(1, buffer_b);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_scalar_gpu.setArg(0, buffer_y);
			kernel_vec_scalar_gpu.setArg(1, (1 / beta));
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_sub_gpu.setArg(0, buffer_res);
			kernel_vec_sub_gpu.setArg(1, buffer_y);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A_t);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res2);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, n);
			kernel_mat_vec_mul_gpu.setArg(5, m);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(n, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_x);
			kernel_vec_sub_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(n));

			kernel_shrink_gpu.setArg(0, buffer_x);
			kernel_shrink_gpu.setArg(1, (tau / beta));
			queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(n));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_x);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res2);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, m);
			kernel_mat_vec_mul_gpu.setArg(5, n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_b);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_scalar_gpu.setArg(0, buffer_res2);
			kernel_vec_scalar_gpu.setArg(1, (gamma * beta));
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_scalar_gpu.setArg(0, buffer_y);
			kernel_vec_scalar_gpu.setArg(1, 1 / (1 / beta));
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_sub_gpu.setArg(0, buffer_y);
			kernel_vec_sub_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));
		}

		queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
		queue.finish();

		vector<float>aux(x.begin(), x.begin() + old_n);
		sol = aux;
		is_solved = true;
		vector<float> err = x;
		mat_vec_mul_avx(A, err, this->T);
		vec_sub_avx(err, b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	vector<float> solution_cpu()
	{
		//	checks if the solution was already computed 
		if (!is_solved)
			solve_cpu();
		return sol;
	}

	vector<float> solution_gpu()
	{
		if (!is_solved)
			solve_gpu();
		return sol;
	}
};

class fista : public sparse
{
public:

	int iterations = 0;
	bool is_solved = false;
	float lambda = 0.3e-2;
	float miu = 1;
	float miu_old = miu;
	float L = 0.0f;

	fista(string A_name, string b_name, float lambda, int iterations) :sparse(A_name, b_name)
	{
		//	the bigger lambda is the more exact the solution should get   
		//	the smaller it is the more likely it is to converge towards a sparse solution 
		this->iterations = iterations;
		this->lambda = lambda;
	}

	void solve_cpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!is_max_eig_set)
			max_eig = power_method_cpu();

		L = 1 / max_eig;

		vector<float>aux1(n), aux2(n), aux3(n);
		mat_scalar_avx(A_t, L, this->T);

		//	Implementing the following FISTA algorithm
		//
		//	Input: : Lipschitz constant L, ?, ? = 1, dictionary A, measurement b, x = 0
		//	While not converge
		//	x(k)?shrink(x(k) - 1 / L A'(Ax(k)-b),?)
		//	?(k + 1)?(1 + ?(1 + 4(?(k) ^ 2)) / 2
		//	x(k)?x(k) + (?(k) - 1) / ?(k + 1) * (x(k) - x(k - 1))
		//	end while
		//	Output: x(k)

		for (int i = 0; i <= iterations; i++)
		{
			aux1 = x;
			aux2 = x;
			miu_old = miu;

			mat_vec_mul_avx(A, aux1, this->T);
			vec_sub_avx(aux1, b, this->T);
			mat_vec_mul_avx(A_t, aux1, this->T);
			vec_sub_avx(x, aux1, this->T);

			aux1 = x;

			miu = (1 + sqrt(1 + 4 * (miu * miu))) / 2;
			vec_sub_avx(aux1, aux2, this->T);
			vec_scalar_avx(aux1, (miu_old - 1) / miu, this->T);
			vec_add_avx(aux1, x, this->T);

			shrink(aux1, lambda, this->T);

			x = aux1;
		}

		vector<float>aux(x.begin(), x.begin() + old_n);
		sol = aux;
		is_solved = true;
		vector<float> err = x;
		mat_vec_mul_avx(A, err, this->T);
		vec_sub_avx(err, b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	void solve_gpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!is_max_eig_set)
			max_eig = power_method_gpu();

		L = 1 / max_eig;

		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		cl::Platform default_platform = all_platforms[0];
		std::vector<cl::Device> all_devices;
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
		cl::Device default_device = all_devices[0];
		cl::Context context(default_device);
		std::ifstream src("gpu_kernels.cl");
		std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
		cl::Program::Sources sources;
		sources.push_back({ str.c_str(),str.length() });
		cl::Program program(context, sources);
		program.build({ default_device });

		vector<float> flat_A;
		vector<float> flat_A_t;
		int t = 32;
		int m1 = 32;

		for (unsigned int i = 0; i < A_t.size(); ++i)
			for (unsigned int j = 0; j < A_t[0].size(); ++j)
			{
				flat_A.push_back(A_t[i][j]);
			}

		for (unsigned int i = 0; i < A.size(); ++i)
			for (unsigned int j = 0; j < A[0].size(); ++j)
			{
				flat_A_t.push_back(A[i][j]);
			}

		vec_scalar_avx(flat_A_t, L, this->T);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A.size());
		cl::Buffer buffer_res1(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res2(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res3(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A_t.size());
		cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(float) * b.size());
		cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());

		cl::CommandQueue queue(context, default_device);

		cl::Kernel kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu");
		cl::Kernel kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu");
		cl::Kernel kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu");
		cl::Kernel kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu");
		cl::Kernel kernel_vec_add_gpu = cl::Kernel(program, "vec_add_gpu");

		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(float) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(float) * b.size(), b.data());
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());

		for (int i = 0; i <= iterations; i++)
		{
			queue.enqueueCopyBuffer(buffer_x, buffer_res3, 0, 0, sizeof(float) * x.size(), NULL, NULL);

			miu_old = miu;

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_x);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, m);
			kernel_mat_vec_mul_gpu.setArg(5, n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res1);
			kernel_vec_sub_gpu.setArg(1, buffer_b);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A_t);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res2);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, n);
			kernel_mat_vec_mul_gpu.setArg(5, m);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(n, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_x);
			kernel_vec_sub_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(n));

			queue.enqueueCopyBuffer(buffer_x, buffer_res1, 0, 0, sizeof(float) * x.size(), NULL, NULL);

			miu = (1 + sqrt(1 + 4 * (miu * miu))) / 2;

			kernel_vec_sub_gpu.setArg(0, buffer_res1);
			kernel_vec_sub_gpu.setArg(1, buffer_res3);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(n));

			kernel_vec_scalar_gpu.setArg(0, buffer_res1);
			kernel_vec_scalar_gpu.setArg(1, (miu_old - 1) / miu);
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(n));

			kernel_vec_add_gpu.setArg(0, buffer_res1);
			kernel_vec_add_gpu.setArg(1, buffer_x);
			queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(n));

			kernel_shrink_gpu.setArg(0, buffer_res1);
			kernel_shrink_gpu.setArg(1, lambda);
			queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(n));

			queue.enqueueCopyBuffer(buffer_res1, buffer_x, 0, 0, sizeof(float) * x.size(), NULL, NULL);
		}

		queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
		queue.finish();

		vector<float>aux(x.begin(), x.begin() + old_n);
		sol = aux;
		is_solved = true;
		vector<float> err = x;
		mat_vec_mul_avx(A, err, this->T);
		vec_sub_avx(err, b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	vector<float> solution_cpu()
	{
		if (!is_solved)
			solve_cpu();
		return sol;
	}

	vector<float> solution_gpu()
	{
		if (!is_solved)
			solve_gpu();
		return sol;
	}
};

class palm : public sparse
{
public:

	int iterations_outter_loop = 0;
	int iterations_inner_loop = 0;
	bool is_solved = false;
	float zeta = 0.001f;
	float t_alg = 1.0;
	float L = 0.0f;

	palm(string A_name, string b_name, int iterations_outter_loop, int iterations_inner_loop) :sparse(A_name, b_name)
	{
		//	the inner loop tipically needs to be a lot smaller than the outter loop 
		//	i.e 100 for the outter one and 3 for the inner one
		this->iterations_outter_loop = iterations_outter_loop;
		this->iterations_inner_loop = iterations_inner_loop;
	}

	void solve_cpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!is_max_eig_set)
			max_eig = power_method_cpu();

		L = max_eig;

		vector<float>aux1(m), aux2(m), aux3(m), aux4(m);
		vector<float> e(m);
		vector<float> w(n);
		vector<float> w1(n);
		vector<float> z(n);
		vector<float> theta(m);
		mat_scalar_avx(A_t, 1 / L, this->T);

		//	Input: Lipschitz constant L, ?, dictionary A, measurement b, x = 0
		//
		//	While not converge(i = 1, 2 …) execute
		//	e(k + 1)?shrink(b - Ax(k) + 1 / ? ?(k), 1 / ?)
		//	t(1)?1, z(1)?x(k), w(1)?x(k)
		//		While not converge(j = 1, 2 …) execute
		//		w(j + 1)?shrink(z(j) + 1 / L A'*(b-A*z(l)-e(k+1)+1/? ?(k)),1/?L)
		//		t(j + 1)?1 / 2 1 + ?(1 + 4t(j) ^ 2)
		//		z(j + 1)?w(j + 1) + (t(j) - 1) / (t(j) + 1)(w(j + 1) - w(j))
		//		Sf - cât timp
		//	x(k + 1)?w(l), ?(k + 1)??(k) + ?(b - Ax(k + 1) - e(k + 1))
		//	end while
		//	Output: x(k)

		for (int i = 0; i < iterations_outter_loop; i++)
		{
			aux1 = x;
			aux2 = b;
			aux3 = theta;
			vec_scalar_avx(aux3, (1 / zeta), this->T);
			mat_vec_mul_avx(A, aux1, this->T);
			vec_sub_avx(aux2, aux1, this->T);
			vec_add_avx(aux2, aux3, this->T);
			shrink(aux2, (1 / zeta), this->T);
			e = aux2;

			t_alg = 1;
			z = x;
			w = x;
			w1 = w;

			for (int j = 0; j < iterations_inner_loop; j++)
			{
				w1 = w;
				aux1 = theta;
				aux2 = b;
				aux3 = z;
				aux4 = z;
				vec_scalar_avx(aux1, (1 / zeta), this->T);
				mat_vec_mul_avx(A, aux3, this->T);
				vec_sub_avx(aux2, aux3, this->T);
				vec_add_avx(aux2, aux1, this->T);
				mat_vec_mul_avx(A_t, aux2, this->T);
				vec_add_avx(aux4, aux2, this->T);
				w = aux4;

				shrink(w, (1 / (zeta * L)), this->T);

				t_alg = (1.0f / 2.0f) * (1 + sqrt(1.0f + 4.0f * t_alg * t_alg));

				aux1 = w;
				aux2 = w;
				vec_sub_avx(aux1, w1, this->T);
				vec_scalar_avx(aux1, (t_alg - 1) / (t_alg + 1), this->T);
				vec_add_avx(aux2, aux1, this->T);
				z = aux2;
			}

			x = w1;

			aux1 = x;
			aux2 = b;

			mat_vec_mul_avx(A, aux1, this->T);
			vec_sub_avx(aux2, aux1, this->T);
			vec_sub_avx(aux2, e, this->T);
			vec_scalar_avx(aux2, zeta, this->T);
			vec_add_avx(theta, aux2, this->T);
		}

		vector<float>aux(x.begin(), x.begin() + old_n);
		sol = aux;
		is_solved = true;
		vector<float> err = x;
		mat_vec_mul_avx(A, err, this->T);
		vec_sub_avx(err, b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	void solve_gpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!is_max_eig_set)
			max_eig = power_method_gpu();

		L = max_eig;

		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		cl::Platform default_platform = all_platforms[0];
		std::vector<cl::Device> all_devices;
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
		cl::Device default_device = all_devices[0];
		cl::Context context(default_device);
		std::ifstream src("gpu_kernels.cl");
		std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
		cl::Program::Sources sources;
		sources.push_back({ str.c_str(),str.length() });
		cl::Program program(context, sources);
		program.build({ default_device });

		vector<float>aux1(m), aux2(m), aux3(m), aux4(m);
		vector<float> e(m);
		vector<float> w(n);
		vector<float> w1(n);
		vector<float> z(n);
		vector<float> theta(m);
		vector<float> flat_A;
		vector<float> flat_A_t;
		int t = 32;
		int m1 = 32;

		for (unsigned int i = 0; i < A_t.size(); ++i)
			for (unsigned int j = 0; j < A_t[0].size(); ++j)
			{
				flat_A.push_back(A_t[i][j]);
			}

		for (unsigned int i = 0; i < A.size(); ++i)
			for (unsigned int j = 0; j < A[0].size(); ++j)
			{
				flat_A_t.push_back(A[i][j]);
			}

		vec_scalar_avx(flat_A_t, 1 / L, this->T);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A.size());
		cl::Buffer buffer_res1(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res2(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res3(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res4(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_res5(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_w(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_w1(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(float) * flat_A_t.size());
		cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(float) * b.size());
		cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_z(context, CL_MEM_READ_WRITE, sizeof(float) * x.size());
		cl::Buffer buffer_e(context, CL_MEM_READ_WRITE, sizeof(float) * b.size());
		cl::Buffer buffer_theta(context, CL_MEM_READ_WRITE, sizeof(float) * theta.size());

		cl::CommandQueue queue(context, default_device);

		cl::Kernel kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu");
		cl::Kernel kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu");
		cl::Kernel kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu");
		cl::Kernel kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu");
		cl::Kernel kernel_vec_add_gpu = cl::Kernel(program, "vec_add_gpu");

		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(float) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(float) * b.size(), b.data());
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
		queue.enqueueWriteBuffer(buffer_e, CL_TRUE, 0, sizeof(float) * b.size(), b.data());

		for (int i = 0; i < iterations_outter_loop; i++)
		{
			queue.enqueueWriteBuffer(buffer_res1, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
			queue.enqueueWriteBuffer(buffer_res2, CL_TRUE, 0, sizeof(float) * b.size(), b.data());
			queue.enqueueWriteBuffer(buffer_res3, CL_TRUE, 0, sizeof(float) * theta.size(), theta.data());

			kernel_vec_scalar_gpu.setArg(0, buffer_res3);
			kernel_vec_scalar_gpu.setArg(1, 1 / zeta);
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(m));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res4);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, m);
			kernel_mat_vec_mul_gpu.setArg(5, n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_res4);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(n));

			kernel_vec_add_gpu.setArg(0, buffer_res2);
			kernel_vec_add_gpu.setArg(1, buffer_res3);
			queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(n));

			kernel_shrink_gpu.setArg(0, buffer_res2);
			kernel_shrink_gpu.setArg(1, 1 / zeta);
			queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(m));

			t_alg = 1;

			queue.enqueueCopyBuffer(buffer_res2, buffer_e, 0, 0, sizeof(float) * e.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_x, buffer_z, 0, 0, sizeof(float) * x.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_x, buffer_w, 0, 0, sizeof(float) * x.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_w, buffer_w1, 0, 0, sizeof(float) * x.size(), NULL, NULL);

			for (int j = 0; j < iterations_inner_loop; j++)
			{
				queue.enqueueCopyBuffer(buffer_w, buffer_w1, 0, 0, sizeof(float) * w.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_theta, buffer_res1, 0, 0, sizeof(float) * theta.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_b, buffer_res2, 0, 0, sizeof(float) * b.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_z, buffer_res3, 0, 0, sizeof(float) * z.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_z, buffer_res4, 0, 0, sizeof(float) * z.size(), NULL, NULL);

				kernel_vec_scalar_gpu.setArg(0, buffer_res1);
				kernel_vec_scalar_gpu.setArg(1, 1 / zeta);
				queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(m));

				kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
				kernel_mat_vec_mul_gpu.setArg(1, buffer_res3);
				kernel_mat_vec_mul_gpu.setArg(2, buffer_res5);
				kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
				kernel_mat_vec_mul_gpu.setArg(4, m);
				kernel_mat_vec_mul_gpu.setArg(5, n);
				queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));

				kernel_vec_sub_gpu.setArg(0, buffer_res2);
				kernel_vec_sub_gpu.setArg(1, buffer_res5);
				queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

				kernel_vec_add_gpu.setArg(0, buffer_res2);
				kernel_vec_add_gpu.setArg(1, buffer_res1);
				queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(m));

				kernel_mat_vec_mul_gpu.setArg(0, buffer_A_t);
				kernel_mat_vec_mul_gpu.setArg(1, buffer_res2);
				kernel_mat_vec_mul_gpu.setArg(2, buffer_res5);
				kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
				kernel_mat_vec_mul_gpu.setArg(4, n);
				kernel_mat_vec_mul_gpu.setArg(5, m);
				queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(n, t), cl::NDRange(m1, t));

				kernel_vec_add_gpu.setArg(0, buffer_res4);
				kernel_vec_add_gpu.setArg(1, buffer_res5);
				queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(n));

				queue.enqueueCopyBuffer(buffer_res4, buffer_w, 0, 0, sizeof(float) * w.size(), NULL, NULL);

				kernel_shrink_gpu.setArg(0, buffer_w);
				kernel_shrink_gpu.setArg(1, (1 / (zeta * L)));
				queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(n));

				t_alg = (1.0f / 2.0f) * (1 + sqrt(1.0f + 4.0f * t_alg * t_alg));

				queue.enqueueCopyBuffer(buffer_w, buffer_res1, 0, 0, sizeof(float) * w.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_w, buffer_res2, 0, 0, sizeof(float) * w.size(), NULL, NULL);

				kernel_vec_sub_gpu.setArg(0, buffer_res1);
				kernel_vec_sub_gpu.setArg(1, buffer_w1);
				queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(n));

				kernel_vec_scalar_gpu.setArg(0, buffer_res1);
				kernel_vec_scalar_gpu.setArg(1, (t_alg - 1.0f) / (t_alg + 1.0f));
				queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(n));

				kernel_vec_add_gpu.setArg(0, buffer_res2);
				kernel_vec_add_gpu.setArg(1, buffer_res1);
				queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(n));

				queue.enqueueCopyBuffer(buffer_res2, buffer_z, 0, 0, sizeof(float) * w.size(), NULL, NULL);
			}

			queue.enqueueCopyBuffer(buffer_w1, buffer_x, 0, 0, sizeof(float) * w.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_x, buffer_res1, 0, 0, sizeof(float) * x.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_b, buffer_res2, 0, 0, sizeof(float) * b.size(), NULL, NULL);

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res3);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, m);
			kernel_mat_vec_mul_gpu.setArg(5, n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_res3);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_e);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_scalar_gpu.setArg(0, buffer_res2);
			kernel_vec_scalar_gpu.setArg(1, zeta);
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(m));

			kernel_vec_add_gpu.setArg(0, buffer_theta);
			kernel_vec_add_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(m));
		}

		queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x.size(), x.data());
		queue.finish();

		vector<float>aux(x.begin(), x.begin() + old_n);
		sol = aux;
		is_solved = true;
		vector<float> err = x;
		mat_vec_mul_avx(A, err, this->T);
		vec_sub_avx(err, b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	vector<float> solution_cpu()
	{
		if (!is_solved)
			solve_cpu();
		return sol;
	}

	vector<float> solution_gpu()
	{
		if (!is_solved)
			solve_gpu();
		return sol;
	}
};