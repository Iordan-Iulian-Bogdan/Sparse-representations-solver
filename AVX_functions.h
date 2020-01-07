#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <array>
#include <immintrin.h>
#include <random>

using namespace std::chrono;
using namespace std;

inline void vec_rand(vector<float>& vec)
{
	std::random_device e;
	std::default_random_engine generator(e());
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	static std::uniform_real_distribution<> dis(-2, 2);
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		vec[i] = dis(generator);
	}
}

inline void vec_rand(vector<double>& vec)
{
	std::random_device e;
	std::default_random_engine generator(e());
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	static std::uniform_real_distribution<> dis(-2, 2);
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		vec[i] = dis(generator);
	}
}

void mat_rand(vector<vector<float>>& mat, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int32_t i = 0; i < mat.size(); i++)
	{
		vec_rand(mat[i]);
	}
}

void mat_rand(vector<vector<double>>& mat, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int32_t i = 0; i < mat.size(); i++)
	{
		vec_rand(mat[i]);
	}
}

void vec_fill(vector<float>& vec, const float a)
{
	__m256 scalar = _mm256_set_ps(a, a, a, a, a, a, a, a);
	for (int i = 0; i < vec.size(); i = i + 8)
	{
		_mm256_storeu_ps(&vec[i], scalar);
	}
}

void vec_fill(vector<double>& vec, const float a)
{
	__m256d scalar = _mm256_set_pd(a, a, a, a);
	for (int i = 0; i < vec.size(); i = i + 4)
	{
		_mm256_storeu_pd(&vec[i], scalar);
	}
}

void mat_fill(vector<vector<float>>& mat, const float a, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
		vec_fill(mat[i], a);
}

void mat_fill(vector<vector<double>>& mat, const float a, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
		vec_fill(mat[i], a);
}

void sparse_vec(vector<float>& vec, const float n)
{
	vec_fill(vec, 0.0f);
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis1(-2, 2);
	static std::uniform_int_distribution<> dis2(0, vec.size());

	for (int i = 0; i < n; i++)
	{
		vec[dis2(e)] = dis1(e);
	}
}

void sparse_vec(vector<double>& vec, const float n)
{
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis1(-2, 2);
	static std::uniform_int_distribution<> dis2(0, vec.size());

	for (int i = 0; i < n; i++)
	{
		vec[dis2(e)] = dis1(e);
	}
}

//	prints a vector
void vec_print(vector<float>& vec)
{
	std::cout << '\n';
	for (unsigned int i = 0; i < vec.size(); i++)
		std::cout << std::right << std::setw(10) << vec[i] << '\n';
	std::cout << '\n';
}

void vec_print(vector<double>& vec)
{
	std::cout << '\n';
	for (unsigned int i = 0; i < vec.size(); i++)
		std::cout << std::right << std::setw(10) << vec[i] << '\n';
	std::cout << '\n';
}

//	prints a matrix 
void mat_print(vector<vector<double>>& mat)
{
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat[0].size(); j++)
		{
			//std::cout << std::right << std::setw(10) << mat[i][j];
			std::cout << mat[i][j] << ' ';
		}
		std::cout << '\n';
	}
}

void mat_print(vector<vector<float>>& mat)
{
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat[0].size(); j++)
		{
			std::cout << std::right << std::setw(10) << mat[i][j];
		}
		std::cout << '\n';
	}
}

//	returns the sign of a float
float sign(float f) {
	if (f > 0.0f)
		return 1;
	return (f == 0) ? 0 : -1;
}

float max_val(float a, float b)
{
	if (a > b)
		return a;
	else
		return b;
}

double sign(double f) {
	if (f > 0.0f)
		return 1;
	return (f == 0) ? 0 : -1;
}

double max_val(double a, double b)
{
	if (a > b)
		return a;
	else
		return b;
}

//	computes the multiplication between a vector "vec" and a scalar "a"
inline void vec_scalar_avx(vector<float>& vec, float a, uint32_t T)
{
	__m256 scalar = _mm256_set_ps(a, a, a, a, a, a, a, a);
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec.size(); i = i + 8)
		_mm256_storeu_ps(&vec[i], _mm256_mul_ps(_mm256_loadu_ps(&vec[i]), scalar));
}

inline void vec_scalar_avx(vector<double>& vec, double a, uint32_t T)
{
	__m256d scalar = _mm256_set_pd(a, a, a, a);
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec.size(); i = i + 4)
		_mm256_storeu_pd(&vec[i], _mm256_mul_pd(_mm256_loadu_pd(&vec[i]), scalar));
}

//	computes the multiplication between a matrix "mat" and a scalar "a"
void mat_scalar_avx(vector<vector<float>>& mat, float a, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
		vec_scalar_avx(mat[i], a, T);
}

void mat_scalar_avx(vector<vector<double>>& mat, double a, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
		vec_scalar_avx(mat[i], a, T);
}

//	substracts vector "vec2" from "vec1" and stores the result in "vec1"
void vec_sub_avx(vector<float>& vec1, vector<float>& vec2, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec1.size(); i = i + 8)
		_mm256_storeu_ps(&vec1[i], _mm256_sub_ps(_mm256_loadu_ps(&vec1[i]), _mm256_loadu_ps(&vec2[i])));
}

void vec_sub_avx(vector<double>& vec1, vector<double>& vec2, uint32_t T)
{
#	pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec1.size(); i = i + 4)
		_mm256_storeu_pd(&vec1[i], _mm256_sub_pd(_mm256_loadu_pd(&vec1[i]), _mm256_loadu_pd(&vec2[i])));
}

//	adds vector "vec2" from "vec1" and stores the result in "vec1"
void vec_add_avx(vector<float>& vec1, vector<float>& vec2, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec1.size(); i = i + 8)
		_mm256_storeu_ps(&vec1[i], _mm256_add_ps(_mm256_loadu_ps(&vec1[i]), _mm256_loadu_ps(&vec2[i])));
}

void vec_add_avx(vector<double>& vec1, vector<double>& vec2, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec1.size(); i = i + 4)
		_mm256_storeu_pd(&vec1[i], _mm256_add_pd(_mm256_loadu_pd(&vec1[i]), _mm256_loadu_pd(&vec2[i])));
}

//	adds all the numbers from a vector (uses only AVX vectorization and is not multithreaded)
inline float reduction_avx(vector<float>& vec)
{
	float sum = 0.0f;
	__m256 aux_avx = _mm256_setzero_ps();
	__m256 sum_avx = _mm256_setzero_ps();

	for (int i = 0; i < vec.size(); i = i + 8)
	{
		aux_avx = _mm256_loadu_ps(&vec[i]);
		sum_avx = _mm256_add_ps(sum_avx, aux_avx);
	}

	//	sum_avx holds the last 8 numbers that we have to add up to get the final answer
	//	we need to use SSE instrinsics since no AVX equivalent exists for a full horizontal add
	sum_avx = _mm256_hadd_ps(sum_avx, sum_avx);
	sum_avx = _mm256_hadd_ps(sum_avx, sum_avx);
	__m128 acc1 = _mm256_extractf128_ps(sum_avx, 0);
	__m128 acc2 = _mm256_extractf128_ps(sum_avx, 1);
	acc1 = _mm_add_ss(acc1, acc2);
	_mm_store_ss(&sum, acc1);

	return sum;
}

inline double reduction_avx(vector<double>& vec)
{
	double sum = 0.0f;
	__m256d aux_avx = _mm256_setzero_pd();
	__m256d sum_avx = _mm256_setzero_pd();

	for (int i = 0; i < vec.size(); i = i + 4)
	{
		aux_avx = _mm256_loadu_pd(&vec[i]);
		sum_avx = _mm256_add_pd(sum_avx, aux_avx);
	}

	//	sum_avx holds the last 8 numbers that we have to add up to get the final answer
	//	we need to use SSE instrinsics since no AVX equivalent exists for a full horizontal add
	sum_avx = _mm256_hadd_pd(sum_avx, sum_avx);
	__m128d acc1 = _mm256_extractf128_pd(sum_avx, 0);
	__m128d acc2 = _mm256_extractf128_pd(sum_avx, 1);
	acc1 = _mm_add_sd(acc1, acc2);
	_mm_store_sd(&sum, acc1);

	return sum;
}

//	multiplies each element from vector "vec1" with the corresponding element in "vec2"  
//	returns a vector which contains these computations
//	is used in conjuction with reduction_avx to compute the dot product between two vectors\
//	(uses only AVX vectorization and is not multithreaded)
inline vector<float> vec_mul_avx(vector<float>& vec1, vector<float>& vec2)
{
	__m256 vec1_avx = _mm256_setzero_ps();
	__m256 vec2_avx = _mm256_setzero_ps();
	__m256 res_avx = _mm256_setzero_ps();
	std::vector<float> vec_aux(vec1.size());

	if (vec1.size() == vec2.size())
	{
		for (int i = 0; i < vec1.size(); i = i + 8)
		{
			vec1_avx = _mm256_loadu_ps(&vec1[i]);
			vec2_avx = _mm256_loadu_ps(&vec2[i]);
			res_avx = _mm256_mul_ps(vec1_avx, vec2_avx);
			_mm256_storeu_ps(&vec_aux[i], res_avx);
		}
	}
	return vec_aux;
}

inline vector<double> vec_mul_avx(vector<double>& vec1, vector<double>& vec2)
{
	__m256d vec1_avx = _mm256_setzero_pd();
	__m256d vec2_avx = _mm256_setzero_pd();
	__m256d res_avx = _mm256_setzero_pd();
	std::vector<double> vec_aux(vec1.size());

	if (vec1.size() == vec2.size())
	{
		for (int i = 0; i < vec1.size(); i = i + 4)
		{
			vec1_avx = _mm256_loadu_pd(&vec1[i]);
			vec2_avx = _mm256_loadu_pd(&vec2[i]);
			res_avx = _mm256_mul_pd(vec1_avx, vec2_avx);
			_mm256_storeu_pd(&vec_aux[i], res_avx);
		}
	}
	return vec_aux;
}

//	computes the dot product between two vectors vec1 and vec2 
//	(uses only AVX vectorization and is not multithreaded)
inline float dot_product_avx(vector<float>& vec1, vector<float>& vec2)
{

	__m256 res_avx = _mm256_setzero_ps();
	float sum = 0.0f;
	std::vector<float> vec_aux(8);

	for (int i = 0; i < vec1.size(); i = i + 8)
		res_avx = _mm256_fmadd_ps(_mm256_loadu_ps(&vec1[i]), _mm256_loadu_ps(&vec2[i]), res_avx);

	//	we have to sum up the elements left in res_avx to get the final answer 
	//	we need to use SSE instrinsics since no AVX equivalent exists for a full horizontal add
	_mm256_storeu_ps(&vec_aux[0], res_avx);
	res_avx = _mm256_hadd_ps(res_avx, res_avx);
	res_avx = _mm256_hadd_ps(res_avx, res_avx);
	__m128 acc1 = _mm256_extractf128_ps(res_avx, 0);
	__m128 acc2 = _mm256_extractf128_ps(res_avx, 1);
	acc1 = _mm_add_ss(acc1, acc2);
	_mm_store_ss(&sum, acc1);

	return sum;
}

inline double dot_product_avx(const vector<double>& vec1, const vector<double>& vec2)
{
	__m256d res_avx = _mm256_setzero_pd();
	double sum = 0.0f;
	std::vector<double> vec_aux(4);

	for (int i = 0; i < vec1.size(); i = i + 4)
		res_avx = _mm256_fmadd_pd(_mm256_loadu_pd(&vec1[i]), _mm256_loadu_pd(&vec2[i]), res_avx);

	//	we have to sum up the elements left in res_avx to get the final answer 
	//	we need to use SSE instrinsics since no AVX equivalent exists for a full horizontal add
	_mm256_storeu_pd(&vec_aux[0], res_avx);
	res_avx = _mm256_hadd_pd(res_avx, res_avx);
	__m128d acc1 = _mm256_extractf128_pd(res_avx, 0);
	__m128d acc2 = _mm256_extractf128_pd(res_avx, 1);
	acc1 = _mm_add_sd(acc1, acc2);
	_mm_store_sd(&sum, acc1);

	return sum;
}

//	computes the multiplication between mat and vec and writes the answer in vec
inline void mat_vec_mul_avx(vector<vector<float>>& mat, vector<float>& vec, uint32_t T)
{
	std::vector<float> res(mat.size());

	//	multiple dot products will be computed in parallel, the dot products themselves are computed sequentially however  
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
	{
		res[i] = dot_product_avx(mat[i], vec);
	}
	vec = res;
}

inline void mat_vec_mul_avx(vector<vector<double>>& mat, vector<double>& vec, uint32_t T)
{
	std::vector<double> res(mat.size());

	//	multiple dot products will be computed in parallel, the dot products themselves are computed sequentially however  
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
	{
		res[i] = dot_product_avx(mat[i], vec);
	}
	vec = res;
}

//	computes the multiplication between two matrices mat1 and mat2 and writes the answer in matrix res
inline void mat_mat_mul_avx(vector<vector<float>>& mat1, vector<vector<float>>& mat2, vector<vector<float>>& res, uint32_t T)
{
	//	multiple dot products will be computed in parallel, the dot products themselves are computed sequentially however  
#pragma omp parallel for num_threads(T) schedule(dynamic) 
	for (int i = 0; i < mat1.size(); i++)
	{
		for (int j = 0; j < mat1.size(); j++)
		{
			res[i][j] = dot_product_avx(mat1[i], mat2[j]);
		}
	}
}

inline void mat_mat_mul_avx(vector<vector<double>>& mat1, vector<vector<double>>& mat2, vector<vector<double>>& res, uint32_t T)
{
	//	multiple dot products will be computed in parallel, the dot products themselves are computed sequentially however  
#pragma omp parallel for num_threads(T) schedule(dynamic) 
	for (int i = 0; i < mat1.size(); i++)
	{
		for (int j = 0; j < mat1.size(); j++)
		{
			res[i][j] = dot_product_avx(mat1[i], mat2[j]);
		}
	}
}

// computes the transpose of mat and writes it to mat_t 
void mat_transpose(vector<vector<float>>& mat, vector<vector<float>>& mat_t, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[0].size(); j++)
			mat_t[j][i] = mat[i][j];
	}
}

void mat_transpose(vector<vector<double>>& mat, vector<vector<double>>& mat_t, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[0].size(); j++)
			mat_t[j][i] = mat[i][j];
	}
}

//	implements the soft thresholding algorithm
void shrink(vector<float>& vec, float threshold, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec.size(); i++)
	{
		vec[i] = sign(vec[i]) * max_val(abs(vec[i]) - threshold, 0);
		if (abs(vec[i]) <= 1.175e-38) //  this is done in order to get rid of "negative zeros" (-0.0f)
			vec[i] = 0.0f;
	}
}

void shrink(vector<double>& vec, double threshold, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec.size(); i++)
	{
		vec[i] = sign(vec[i]) * max_val(abs(vec[i]) - threshold, 0);
		if (abs(vec[i]) <= 1.175e-38) //  this is done in order to get rid of "negative zeros" (-0.0f)
			vec[i] = 0.0f;
	}
}

//	computes the L2 norm of a vector
float norm(vector<float>& vec)
{
	float norm = 0.0f;

	for (int i = 0; i < vec.size(); i++)
		norm += vec[i] * vec[i];

	return (float)sqrt(norm);
}

double norm(vector<double>& vec)
{
	double norm = 0.0f;

	for (int i = 0; i < vec.size(); i++)
		norm += vec[i] * vec[i];

	return (double)sqrt(norm);
}

void flatten(vector<vector<double>>& mat, vector<double>& flat_mat, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); ++i)
		for (int j = 0; j < mat[0].size(); j = j + 4)
		{
			_mm256_stream_pd(&flat_mat[i * mat[0].size() + j], _mm256_loadu_pd(&mat[i][j]));
		}
}

void flatten(vector<vector<float>>& mat, vector<float>& flat_mat, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); ++i)
		for (int j = 0; j < mat[0].size(); j = j + 8)
		{
			_mm256_stream_ps(&flat_mat[i * mat[0].size() + j], _mm256_loadu_ps(&mat[i][j]));
		}
}

void unflatten(vector<vector<float>>& mat, vector<float>& flat_mat, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); ++i)
		for (int j = 0; j < mat[0].size(); j = j + 8)
		{
			_mm256_store_ps(&mat[i][j], _mm256_loadu_ps(&flat_mat[i * mat[0].size() + j]));
		}
}

void unflatten(vector<vector<double>>& mat, vector<double>& flat_mat, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < mat.size(); ++i)
		for (int j = 0; j < mat[0].size(); j = j + 4)
		{
			_mm256_store_pd(&mat[i][j], _mm256_loadu_pd(&flat_mat[i * mat[0].size() + j]));
		}
}

void copy(vector<vector<double>>& destination, vector<vector<double>>& source, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < source.size(); ++i)
		for (int j = 0; j < source[0].size(); ++j)
			destination[i][j] = source[i][j];
}

void copy(vector<vector<float>>& destination, vector<vector<float>>& source, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < source.size(); ++i)
		for (int j = 0; j < source[0].size(); ++j)
			std::copy(source[i].begin(), source[i].end(), destination[i].begin());
}
