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

template <class T_type>
class sparse
{
public:

	uint32_t n = 0;
	uint32_t m = 0;
	uint32_t T = omp_get_num_procs();
	uint32_t old_n = 0;
	uint32_t old_m = 0;
	T_type max_eig = 0.0f;
	T_type error = 0.0f;
	vector<vector<T_type>> A;
	vector<vector<T_type>> A_t;
	vector<T_type> b;
	vector<T_type> x;
	vector<T_type> sol;
	T_type solve_time = 0.0f;
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
		this->A.resize(m, vector<T_type>(n));
		this->A_t.resize(n, vector<T_type>(m));
		this->b.resize(m);
		this->x.resize(n);

		//	filling A and b with zeros before reading into them the actual data
		//	the extra lines and collums will not affect the computation since they are now filled with zeros

		fill(b.begin(), b.end(), 0.0f);

		for (unsigned int i = 0; i < A.size(); ++i)
			fill(A[i].begin(), A[i].end(), 0.0f);

		//	reading the data from disk
		for (unsigned int i = 0; i < A.size(); ++i)
			fin1.read(reinterpret_cast<char*>(A[i].data()), (A[i].size() - (n - old_n)) * sizeof(T_type));
		fin1.close();

		fin2.read(reinterpret_cast<char*>(b.data()), (b.size()) * sizeof(T_type));
		fin2.close();

		mat_transpose(A, A_t, T);
	}

	sparse()
	{}

	//	sets the number of threads that the CPU solver can use 
	//	by default this is set to the maximum number of threads available on the machine
	void set_number_of_threads(int T)
	{
		this->T = T;
	}

	vector<vector<T_type>> get_dictionary()
	{
		return A;
	}

	vector<T_type> get_measurement()
	{
		return b;
	}

	void set_max_eig(T_type max_eig)
	{
		this->max_eig = max_eig;
		is_max_eig_set = true;
	}

	void write_solution_to_disk()
	{
		ofstream fout("solution.bin", ios::out | ios::binary);
		fout.write((char*)&sol[0], sol.size() * sizeof(sol));
		fout.close();
	}

protected:

	//	function which aproximates the maximum singular value using the power method 
	//	(required for the convergence of all three algorithms)
	//	is GPU accelarated (namely the matrix multiplication which is very expensive for large matrices)
	T_type power_method_gpu()
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

		vector<T_type> flat_A;
		vector<T_type> flat_A_t;
		vector<vector<T_type>> X(m, vector<T_type>(m));
		vector<T_type> flat_X;

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
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A_t.size());
		cl::Buffer buffer_X(context, CL_MEM_READ_WRITE, sizeof(T_type) * flat_X.size());

		cl::CommandQueue queue(context, default_device);

		//	compiling the kernel

		cl::Kernel kernel_mat_mat_mul_gpu;
		
		// checking if the format is double or float and we compile the appropriate kernel 
		
		if (sizeof(T_type) == 8)
			kernel_mat_mat_mul_gpu = cl::Kernel(program, "mat_mat_mul_gpu_dp");
		else
			kernel_mat_mat_mul_gpu = cl::Kernel(program, "mat_mat_mul_gpu_sp");

		//	writting tot the buffers
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(T_type) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(T_type) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_X, CL_TRUE, 0, sizeof(T_type) * flat_X.size(), flat_X.data());

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
		queue.enqueueReadBuffer(buffer_X, CL_TRUE, 0, sizeof(T_type) * flat_X.size(), flat_X.data());
		queue.finish();

		uint64_t k = 0;

		//	"unflattening" the data back
		for (unsigned int i = 0; i < X.size(); ++i)
			for (unsigned int j = 0; j < X[0].size(); ++j)
			{
				X[i][j] = flat_X[k++];
			}

		vector<T_type> b_k(m);
		vector<T_type> aux(m);
		vector<T_type> b_k1(m);
		T_type norm_b_k1 = 0.0f;
		T_type eig = 0.0f;

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
	T_type power_method_cpu()
	{
		vector<vector<T_type>> X(m, vector<T_type>(m));
		vector<T_type> b_k(m);
		vector<T_type> aux(m);
		vector<T_type> b_k1(m);
		T_type norm_b_k1 = 0.0f;
		T_type eig = 0.0f;
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

template <class T_type>
class adm : public sparse<T_type>
{
public:

	int iterations = 0;
	bool is_solved = false;
	T_type gamma = 0.0f;
	T_type beta = 0.0f;
	T_type tau = 0.0f;

	adm(string A_name, string b_name, T_type beta, T_type tau, int iterations) :sparse<T_type>(A_name, b_name)
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
		if (!this->is_max_eig_set)
			this->max_eig = this->power_method_cpu();

		//	gamma is required to be less than 2 in order to ensure convergence
		//	note that if the maximum singular value are too big this can no longer be true
		//	and you'd need to change tau and beta
		this->gamma = 1.99f - (this->tau * this->max_eig);

		vector<T_type>aux1(this->n), aux2(this->n), aux3(this->n);
		vector<T_type>y(this->m);
		vector<T_type>r(this->m);
		mat_scalar_avx(this->A_t, tau, this->T);

		//	Implemeting the following ADM algorithm :
		//	Input: τ, β, γ dictionary A, measurement b, x = 0, y = 0
		//	While not converge
		//	x(k)←shrink(x(k)-τA*(Ax(k)-b-y(k)/β),τ/β)
		//	y(k+1)←y(k)-γβ(Ax(k+1)-b)
		//	end while
		//	Output: x(k)

		for (int i = 0; i <= iterations; i++)
		{
			aux1 = this->x;
			aux2 = y;

			mat_vec_mul_avx(this->A, aux1, this->T);
			vec_sub_avx(aux1, this->b, this->T);
			vec_scalar_avx(aux2, (1 / beta), this->T);
			vec_sub_avx(aux1, aux2, this->T);
			mat_vec_mul_avx(this->A_t, aux1, this->T);
			vec_sub_avx(this->x, aux1, this->T);
			shrink(this->x, (tau / beta), this->T);

			aux1 = this->x;
			mat_vec_mul_avx(this->A, aux1, this->T);
			vec_sub_avx(aux1, this->b, this->T);
			vec_scalar_avx(aux1, (gamma * beta), this->T);
			vec_sub_avx(y, aux1, this->T);
		}

		//	the solution is cut down to the correct size, remember that we resized A and b when we read them
		vector<T_type>aux(this->x.begin(), this->x.begin() + this->old_n);
		this->sol = aux;
		is_solved = true;
		vector<T_type> err = this->x;
		mat_vec_mul_avx(this->A, err, this->T);
		vec_sub_avx(err, this->b, this->T);
		this->error = norm(err); //	computing the error A * x - b
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		this->solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	//	same as above except all the computations inside the main loop are now replaced 
	//	with the equivalent OpenCL kernels  
	void solve_gpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!this->is_max_eig_set)
			this->max_eig = this->power_method_gpu();

		this->gamma = 1.99f - (this->tau * this->max_eig);

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

		vector<T_type> flat_A;
		vector<T_type> flat_A_t;
		vector<T_type>y(this->m);
		int t = 32;
		int m1 = 32;

		for (unsigned int i = 0; i < this->A_t.size(); ++i)
			for (unsigned int j = 0; j < this->A_t[0].size(); ++j)
			{
				flat_A.push_back(this->A_t[i][j]);
			}

		for (unsigned int i = 0; i < this->A.size(); ++i)
			for (unsigned int j = 0; j < this->A[0].size(); ++j)
			{
				flat_A_t.push_back(this->A[i][j]);
			}

		vec_scalar_avx(flat_A_t, tau, this->T);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A.size());
		cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_y(context, CL_MEM_READ_WRITE, sizeof(T_type) * y.size());
		cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res2(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A_t.size());
		cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(T_type) * this->b.size());

		cl::CommandQueue queue(context, default_device);

		cl::Kernel kernel_mat_vec_mul_gpu;
		cl::Kernel kernel_vec_sub_gpu;
		cl::Kernel kernel_vec_scalar_gpu;
		cl::Kernel kernel_shrink_gpu;

		if (sizeof(T_type) == 8)
		{
			kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_dp");
			kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu_dp");
			kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu_dp");
			kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu_dp");
		}
		else
		{
			kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_sp");
			kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu_sp");
			kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu_sp");
			kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu_sp");
		}

		queue.enqueueWriteBuffer(buffer_y, CL_TRUE, 0, sizeof(T_type) * y.size(), y.data());
		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(T_type) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(T_type) * this->b.size(), this->b.data());
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(T_type) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_y, CL_TRUE, 0, sizeof(T_type) * y.size(), y.data());
		queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());

		for (int i = 0; i <= iterations; i++)
		{
			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_x);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->m);
			kernel_mat_vec_mul_gpu.setArg(5, this->n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res);
			kernel_vec_sub_gpu.setArg(1, buffer_b);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_scalar_gpu.setArg(0, buffer_y);
			kernel_vec_scalar_gpu.setArg(1, (1 / beta));
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_sub_gpu.setArg(0, buffer_res);
			kernel_vec_sub_gpu.setArg(1, buffer_y);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A_t);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res2);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->n);
			kernel_mat_vec_mul_gpu.setArg(5, this->m);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->n, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_x);
			kernel_vec_sub_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_shrink_gpu.setArg(0, buffer_x);
			kernel_shrink_gpu.setArg(1, (tau / beta));
			queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_x);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res2);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->m);
			kernel_mat_vec_mul_gpu.setArg(5, this->n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_b);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_scalar_gpu.setArg(0, buffer_res2);
			kernel_vec_scalar_gpu.setArg(1, (gamma * beta));
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_scalar_gpu.setArg(0, buffer_y);
			kernel_vec_scalar_gpu.setArg(1, 1 / (1 / beta));
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_sub_gpu.setArg(0, buffer_y);
			kernel_vec_sub_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));
		}

		queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());
		queue.finish();

		vector<T_type>aux(this->x.begin(), this->x.begin() + this->old_n);
		this->sol = aux;
		is_solved = true;
		vector<T_type> err = this->x;
		mat_vec_mul_avx(this->A, err, this->T);
		vec_sub_avx(err, this->b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		this->solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	vector<T_type> solution_cpu()
	{
		//	checks if the solution was already computed 
		if (!is_solved)
			solve_cpu();
		return this->sol;
	}

	vector<T_type> solution_gpu()
	{
		if (!is_solved)
			solve_gpu();
		return this->sol;
	}
};

template <class T_type>
class fista : public sparse<T_type>
{
public:

	int iterations = 0;
	bool is_solved = false;
	T_type lambda = 0.3e-2;
	T_type miu = 1;
	T_type miu_old = miu;
	T_type L = 0.0f;

	fista(string A_name, string b_name, T_type lambda, int iterations) :sparse<T_type>(A_name, b_name)
	{
		//	the bigger lambda is the more exact the solution should get   
		//	the smaller it is the more likely it is to converge towards a sparse solution 
		this->iterations = iterations;
		this->lambda = lambda;
	}

	void solve_cpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!this->is_max_eig_set)
			this->max_eig = this->power_method_cpu();

		L = 1 / this->max_eig;

		vector<T_type>aux1(this->n), aux2(this->n), aux3(this->n);
		mat_scalar_avx(this->A_t, L, this->T);

		//	Implementing the following FISTA algorithm
		//
		//	Input: : Lipschitz constant L, λ, μ=1, dictionary A, measurement b, x = 0
		//	While not converge
		//	x(k)←shrink(x(k) - 1 / L A'(Ax(k)-b),λ)
		//	μ(k+1)←(1+√(1+4μ(k)^2 ))/2
		//	x(k)←x(k)+(μ(k)-1)/μ(k+1)*(x(k)-x(k-1))
		//	end while
		//	Output: x(k)

		for (int i = 0; i <= iterations; i++)
		{
			aux1 = this->x;
			aux2 = this->x;
			miu_old = miu;

			mat_vec_mul_avx(this->A, aux1, this->T);
			vec_sub_avx(aux1, this->b, this->T);
			mat_vec_mul_avx(this->A_t, aux1, this->T);
			vec_sub_avx(this->x, aux1, this->T);

			aux1 = this->x;

			miu = (1 + sqrt(1 + 4 * (miu * miu))) / 2;
			vec_sub_avx(aux1, aux2, this->T);
			vec_scalar_avx(aux1, (miu_old - 1) / miu, this->T);
			vec_add_avx(aux1, this->x, this->T);

			shrink(aux1, lambda, this->T);

			this->x = aux1;
		}

		vector<T_type>aux(this->x.begin(), this->x.begin() + this->old_n);
		this->sol = aux;
		is_solved = true;
		vector<T_type> err = this->x;
		mat_vec_mul_avx(this->A, err, this->T);
		vec_sub_avx(err, this->b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		this->solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	void solve_gpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!this->is_max_eig_set)
			this->max_eig = this->power_method_gpu();

		L = 1 / this->max_eig;

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

		vector<T_type> flat_A;
		vector<T_type> flat_A_t;
		int t = 32;
		int m1 = 32;

		for (unsigned int i = 0; i < this->A_t.size(); ++i)
			for (unsigned int j = 0; j < this->A_t[0].size(); ++j)
			{
				flat_A.push_back(this->A_t[i][j]);
			}

		for (unsigned int i = 0; i < this->A.size(); ++i)
			for (unsigned int j = 0; j < this->A[0].size(); ++j)
			{
				flat_A_t.push_back(this->A[i][j]);
			}

		vec_scalar_avx(flat_A_t, L, this->T);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A.size());
		cl::Buffer buffer_res1(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res2(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res3(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A_t.size());
		cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(T_type) * this->b.size());
		cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());

		cl::CommandQueue queue(context, default_device);

		cl::Kernel kernel_mat_vec_mul_gpu;
		cl::Kernel kernel_vec_sub_gpu;
		cl::Kernel kernel_vec_scalar_gpu;
		cl::Kernel kernel_shrink_gpu;
		cl::Kernel kernel_vec_add_gpu;

		if (sizeof(T_type) == 8)
		{
			kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_dp");
			kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu_dp");
			kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu_dp");
			kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu_dp");
			kernel_vec_add_gpu = cl::Kernel(program, "vec_add_gpu_dp");
		}
		else
		{
			kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_sp");
			kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu_sp");
			kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu_sp");
			kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu_sp");
			kernel_vec_add_gpu = cl::Kernel(program, "vec_add_gpu_sp");
		}

		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(T_type) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(T_type) * this->b.size(), this->b.data());
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(T_type) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());

		for (int i = 0; i <= iterations; i++)
		{
			queue.enqueueCopyBuffer(buffer_x, buffer_res3, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);

			miu_old = miu;

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_x);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->m);
			kernel_mat_vec_mul_gpu.setArg(5, this->n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res1);
			kernel_vec_sub_gpu.setArg(1, buffer_b);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A_t);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res2);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->n);
			kernel_mat_vec_mul_gpu.setArg(5, this->m);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->n, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_x);
			kernel_vec_sub_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->n));

			queue.enqueueCopyBuffer(buffer_x, buffer_res1, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);

			miu = (1 + sqrt(1 + 4 * (miu * miu))) / 2;

			kernel_vec_sub_gpu.setArg(0, buffer_res1);
			kernel_vec_sub_gpu.setArg(1, buffer_res3);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_vec_scalar_gpu.setArg(0, buffer_res1);
			kernel_vec_scalar_gpu.setArg(1, (miu_old - 1) / miu);
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_vec_add_gpu.setArg(0, buffer_res1);
			kernel_vec_add_gpu.setArg(1, buffer_x);
			queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_shrink_gpu.setArg(0, buffer_res1);
			kernel_shrink_gpu.setArg(1, lambda);
			queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(this->n));

			queue.enqueueCopyBuffer(buffer_res1, buffer_x, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);
		}

		queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());
		queue.finish();

		vector<T_type>aux(this->x.begin(), this->x.begin() + this->old_n);
		this->sol = aux;
		is_solved = true;
		vector<T_type> err = this->x;
		mat_vec_mul_avx(this->A, err, this->T);
		vec_sub_avx(err, this->b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		this->solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	vector<T_type> solution_cpu()
	{
		if (!is_solved)
			solve_cpu();
		return this->sol;
	}

	vector<T_type> solution_gpu()
	{
		if (!is_solved)
			solve_gpu();
		return this->sol;
	}
};

template <class T_type>
class palm : public sparse<T_type>
{
public:

	int iterations_outter_loop = 0;
	int iterations_inner_loop = 0;
	bool is_solved = false;
	T_type zeta = 0.001f;
	T_type t_alg = 1.0;
	T_type L = 0.0f;

	palm(string A_name, string b_name, int iterations_outter_loop, int iterations_inner_loop) :sparse<T_type>(A_name, b_name)
	{
		//	the inner loop tipically needs to be a lot smaller than the outter loop 
		//	i.e 100 for the outter one and 3 for the inner one
		this->iterations_outter_loop = iterations_outter_loop;
		this->iterations_inner_loop = iterations_inner_loop;
	}

	void solve_cpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!this->is_max_eig_set)
			this->max_eig = this->power_method_cpu();

		L = this->max_eig;

		vector<T_type>aux1(this->m), aux2(this->m), aux3(this->m), aux4(this->m);
		vector<T_type> e(this->m);
		vector<T_type> w(this->n);
		vector<T_type> w1(this->n);
		vector<T_type> z(this->n);
		vector<T_type> theta(this->m);
		mat_scalar_avx(this->A_t, 1 / L, this->T);

		//	Input: Lipschitz constant L, ξ, dictionary A, measurement b, x = 0
		//
		//	While not converge(i = 1, 2 …) execute
		//	e(k+1)←shrink(b-Ax(k)+1/ξθ(k),1/ξ)
		//	t(1)←1, z(1)←x(k), w(1)←x(k)
		//		While not converge(j = 1, 2 …) execute
		//		w(j+1)←shrink( z(j)+1/L A'(b-Az(l)-e(k+1)+1/ξθ(k)),1/ξL)
		//		t(j+1)←1/2 1+√(1+4t(j)^2 )
		//		z(j+1)←w(j+1)+(t(j)-1)/(t(j)+1)(w(j+1)-w(j))
		//		Sf - cât timp
		//	x(k+1)←w(l),θ(k+1)←θ(k)+ξ(b-Ax(k+1)-e(k+1))
		//	end while
		//	Output: x(k)

		for (int i = 0; i < iterations_outter_loop; i++)
		{
			aux1 = this->x;
			aux2 = this->b;
			aux3 = theta;
			vec_scalar_avx(aux3, (1 / zeta), this->T);
			mat_vec_mul_avx(this->A, aux1, this->T);
			vec_sub_avx(aux2, aux1, this->T);
			vec_add_avx(aux2, aux3, this->T);
			shrink(aux2, (1 / zeta), this->T);
			e = aux2;

			t_alg = 1;
			z = this->x;
			w = this->x;
			w1 = w;

			for (int j = 0; j < iterations_inner_loop; j++)
			{
				w1 = w;
				aux1 = theta;
				aux2 = this->b;
				aux3 = z;
				aux4 = z;
				vec_scalar_avx(aux1, (1 / zeta), this->T);
				mat_vec_mul_avx(this->A, aux3, this->T);
				vec_sub_avx(aux2, aux3, this->T);
				vec_add_avx(aux2, aux1, this->T);
				mat_vec_mul_avx(this->A_t, aux2, this->T);
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

			this->x = w1;

			aux1 = this->x;
			aux2 = this->b;

			mat_vec_mul_avx(this->A, aux1, this->T);
			vec_sub_avx(aux2, aux1, this->T);
			vec_sub_avx(aux2, e, this->T);
			vec_scalar_avx(aux2, zeta, this->T);
			vec_add_avx(theta, aux2, this->T);
		}

		vector<T_type>aux(this->x.begin(), this->x.begin() + this->old_n);
		this->sol = aux;
		is_solved = true;
		vector<T_type> err = this->x;
		mat_vec_mul_avx(this->A, err, this->T);
		vec_sub_avx(err, this->b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		this->solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	void solve_gpu()
	{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		if (!this->is_max_eig_set)
			this->max_eig = this->power_method_gpu();

		L = this->max_eig;

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

		vector<T_type>aux1(this->m), aux2(this->m), aux3(this->m), aux4(this->m);
		vector<T_type> e(this->m);
		vector<T_type> w(this->n);
		vector<T_type> w1(this->n);
		vector<T_type> z(this->n);
		vector<T_type> theta(this->m);
		vector<T_type> flat_A;
		vector<T_type> flat_A_t;
		int t = 32;
		int m1 = 32;

		for (unsigned int i = 0; i < this->A_t.size(); ++i)
			for (unsigned int j = 0; j < this->A_t[0].size(); ++j)
			{
				flat_A.push_back(this->A_t[i][j]);
			}

		for (unsigned int i = 0; i < this->A.size(); ++i)
			for (unsigned int j = 0; j < this->A[0].size(); ++j)
			{
				flat_A_t.push_back(this->A[i][j]);
			}

		vec_scalar_avx(flat_A_t, 1 / L, this->T);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A.size());
		cl::Buffer buffer_res1(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res2(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res3(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res4(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_res5(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_w(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_w1(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(T_type) * flat_A_t.size());
		cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(T_type) * this->b.size());
		cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_z(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->x.size());
		cl::Buffer buffer_e(context, CL_MEM_READ_WRITE, sizeof(T_type) * this->b.size());
		cl::Buffer buffer_theta(context, CL_MEM_READ_WRITE, sizeof(T_type) * theta.size());

		cl::CommandQueue queue(context, default_device);

		cl::Kernel kernel_mat_vec_mul_gpu;
		cl::Kernel kernel_vec_sub_gpu;
		cl::Kernel kernel_vec_scalar_gpu;
		cl::Kernel kernel_shrink_gpu;
		cl::Kernel kernel_vec_add_gpu;

		if (sizeof(T_type) == 8)
		{
			kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_dp");
			kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu_dp");
			kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu_dp");
			kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu_dp");
			kernel_vec_add_gpu = cl::Kernel(program, "vec_add_gpu_dp");
		}
		else
		{
			kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_sp");
			kernel_vec_sub_gpu = cl::Kernel(program, "vec_sub_gpu_sp");
			kernel_vec_scalar_gpu = cl::Kernel(program, "vec_scalar_gpu_sp");
			kernel_shrink_gpu = cl::Kernel(program, "shrink_gpu_sp");
			kernel_vec_add_gpu = cl::Kernel(program, "vec_add_gpu_sp");
		}

		queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(T_type) * flat_A_t.size(), flat_A_t.data());
		queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(T_type) * this->b.size(), this->b.data());
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(T_type) * flat_A.size(), flat_A.data());
		queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());
		queue.enqueueWriteBuffer(buffer_e, CL_TRUE, 0, sizeof(T_type) * this->b.size(), this->b.data());

		for (int i = 0; i < iterations_outter_loop; i++)
		{
			queue.enqueueWriteBuffer(buffer_res1, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());
			queue.enqueueWriteBuffer(buffer_res2, CL_TRUE, 0, sizeof(T_type) * this->b.size(), this->b.data());
			queue.enqueueWriteBuffer(buffer_res3, CL_TRUE, 0, sizeof(T_type) * theta.size(), theta.data());

			kernel_vec_scalar_gpu.setArg(0, buffer_res3);
			kernel_vec_scalar_gpu.setArg(1, 1 / zeta);
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res4);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->m);
			kernel_mat_vec_mul_gpu.setArg(5, this->n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_res4);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_vec_add_gpu.setArg(0, buffer_res2);
			kernel_vec_add_gpu.setArg(1, buffer_res3);
			queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(this->n));

			kernel_shrink_gpu.setArg(0, buffer_res2);
			kernel_shrink_gpu.setArg(1, 1 / zeta);
			queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(this->m));

			t_alg = 1;

			queue.enqueueCopyBuffer(buffer_res2, buffer_e, 0, 0, sizeof(T_type) * e.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_x, buffer_z, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_x, buffer_w, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_w, buffer_w1, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);

			for (int j = 0; j < iterations_inner_loop; j++)
			{
				queue.enqueueCopyBuffer(buffer_w, buffer_w1, 0, 0, sizeof(T_type) * w.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_theta, buffer_res1, 0, 0, sizeof(T_type) * theta.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_b, buffer_res2, 0, 0, sizeof(T_type) * this->b.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_z, buffer_res3, 0, 0, sizeof(T_type) * z.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_z, buffer_res4, 0, 0, sizeof(T_type) * z.size(), NULL, NULL);

				kernel_vec_scalar_gpu.setArg(0, buffer_res1);
				kernel_vec_scalar_gpu.setArg(1, 1 / zeta);
				queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->m));

				kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
				kernel_mat_vec_mul_gpu.setArg(1, buffer_res3);
				kernel_mat_vec_mul_gpu.setArg(2, buffer_res5);
				kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
				kernel_mat_vec_mul_gpu.setArg(4, this->m);
				kernel_mat_vec_mul_gpu.setArg(5, this->n);
				queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->m, t), cl::NDRange(m1, t));

				kernel_vec_sub_gpu.setArg(0, buffer_res2);
				kernel_vec_sub_gpu.setArg(1, buffer_res5);
				queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

				kernel_vec_add_gpu.setArg(0, buffer_res2);
				kernel_vec_add_gpu.setArg(1, buffer_res1);
				queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(this->m));

				kernel_mat_vec_mul_gpu.setArg(0, buffer_A_t);
				kernel_mat_vec_mul_gpu.setArg(1, buffer_res2);
				kernel_mat_vec_mul_gpu.setArg(2, buffer_res5);
				kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
				kernel_mat_vec_mul_gpu.setArg(4, this->n);
				kernel_mat_vec_mul_gpu.setArg(5, this->m);
				queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->n, t), cl::NDRange(m1, t));

				kernel_vec_add_gpu.setArg(0, buffer_res4);
				kernel_vec_add_gpu.setArg(1, buffer_res5);
				queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(this->n));

				queue.enqueueCopyBuffer(buffer_res4, buffer_w, 0, 0, sizeof(T_type) * w.size(), NULL, NULL);

				kernel_shrink_gpu.setArg(0, buffer_w);
				kernel_shrink_gpu.setArg(1, (1 / (zeta * L)));
				queue.enqueueNDRangeKernel(kernel_shrink_gpu, cl::NullRange, cl::NDRange(this->n));

				t_alg = (1.0f / 2.0f) * (1 + sqrt(1.0f + 4.0f * t_alg * t_alg));

				queue.enqueueCopyBuffer(buffer_w, buffer_res1, 0, 0, sizeof(T_type) * w.size(), NULL, NULL);
				queue.enqueueCopyBuffer(buffer_w, buffer_res2, 0, 0, sizeof(T_type) * w.size(), NULL, NULL);

				kernel_vec_sub_gpu.setArg(0, buffer_res1);
				kernel_vec_sub_gpu.setArg(1, buffer_w1);
				queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->n));

				kernel_vec_scalar_gpu.setArg(0, buffer_res1);
				kernel_vec_scalar_gpu.setArg(1, (t_alg - 1.0f) / (t_alg + 1.0f));
				queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->n));

				kernel_vec_add_gpu.setArg(0, buffer_res2);
				kernel_vec_add_gpu.setArg(1, buffer_res1);
				queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(this->n));

				queue.enqueueCopyBuffer(buffer_res2, buffer_z, 0, 0, sizeof(T_type) * w.size(), NULL, NULL);
			}

			queue.enqueueCopyBuffer(buffer_w1, buffer_x, 0, 0, sizeof(T_type) * w.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_x, buffer_res1, 0, 0, sizeof(T_type) * this->x.size(), NULL, NULL);
			queue.enqueueCopyBuffer(buffer_b, buffer_res2, 0, 0, sizeof(T_type) * this->b.size(), NULL, NULL);

			kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
			kernel_mat_vec_mul_gpu.setArg(1, buffer_res1);
			kernel_mat_vec_mul_gpu.setArg(2, buffer_res3);
			kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(T_type), NULL);
			kernel_mat_vec_mul_gpu.setArg(4, this->m);
			kernel_mat_vec_mul_gpu.setArg(5, this->n);
			queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(this->m, t), cl::NDRange(m1, t));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_res3);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_sub_gpu.setArg(0, buffer_res2);
			kernel_vec_sub_gpu.setArg(1, buffer_e);
			queue.enqueueNDRangeKernel(kernel_vec_sub_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_scalar_gpu.setArg(0, buffer_res2);
			kernel_vec_scalar_gpu.setArg(1, zeta);
			queue.enqueueNDRangeKernel(kernel_vec_scalar_gpu, cl::NullRange, cl::NDRange(this->m));

			kernel_vec_add_gpu.setArg(0, buffer_theta);
			kernel_vec_add_gpu.setArg(1, buffer_res2);
			queue.enqueueNDRangeKernel(kernel_vec_add_gpu, cl::NullRange, cl::NDRange(this->m));
		}

		queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(T_type) * this->x.size(), this->x.data());
		queue.finish();

		vector<T_type>aux(this->x.begin(), this->x.begin() + this->old_n);
		this->sol = aux;
		is_solved = true;
		vector<T_type> err = this->x;
		mat_vec_mul_avx(this->A, err, this->T);
		vec_sub_avx(err, this->b, this->T);
		this->error = norm(err);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		this->solve_time = duration_cast<milliseconds>(t2 - t1).count();
	}

	vector<T_type> solution_cpu()
	{
		if (!is_solved)
			solve_cpu();
		return this->sol;
	}

	vector<T_type> solution_gpu()
	{
		if (!is_solved)
			solve_gpu();
		return this->sol;
	}
};
