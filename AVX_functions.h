#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <array>
#include <immintrin.h>

using namespace std::chrono;
using namespace std;


//	prints a vector
void vec_print(vector<float>& vec)
{
	std::cout << '\n';
	for (unsigned int i = 0; i < vec.size(); i++)
		std::cout << setprecision(16) << vec[i] << '\n';
	std::cout << '\n';
}

//	prints a matrix 
void mat_print(vector<vector<float>>& mat)
{
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat[0].size(); j++)
			std::cout << mat[i][j] << ' ';
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

//	computes the multiplication between a vector "vec" and a scalar "a"
void vec_scalar_avx(vector<float>& vec, float a, uint32_t T)
{
	__m256 scalar = _mm256_set_ps(a, a, a, a, a, a, a, a);
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec.size(); i = i + 8)
	{
		__m256 res_avx = _mm256_loadu_ps(&vec[i]);
		res_avx = _mm256_mul_ps(res_avx, scalar);
		_mm256_storeu_ps(&vec[i], res_avx);
	}
}

//	computes the multiplication between a matrix "mat" and a scalar "a"
void mat_scalar_avx(vector<vector<float>>& mat, float a, uint32_t T)
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
	{
		__m256 vec1_avx = _mm256_loadu_ps(&vec1[i]);
		__m256 vec2_avx = _mm256_loadu_ps(&vec2[i]);
		__m256 res_avx = _mm256_sub_ps(vec1_avx, vec2_avx);
		_mm256_storeu_ps(&vec1[i], res_avx);
	}
}

//	adds vector "vec2" from "vec1" and stores the result in "vec1"
void vec_add_avx(vector<float>& vec1, vector<float>& vec2, uint32_t T)
{
#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < vec1.size(); i = i + 8)
	{
		__m256 vec1_avx = _mm256_loadu_ps(&vec1[i]);
		__m256 vec2_avx = _mm256_loadu_ps(&vec2[i]);
		__m256 res_avx = _mm256_add_ps(vec1_avx, vec2_avx);
		_mm256_storeu_ps(&vec1[i], res_avx);
	}
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

//	computes the dot product between two vectors vec1 and vec2 
//	(uses only AVX vectorization and is not multithreaded)
inline float dot_product_avx(vector<float>& vec1, vector<float>& vec2)
{
	__m256 vec1_avx = _mm256_setzero_ps();
	__m256 vec2_avx = _mm256_setzero_ps();
	__m256 res_avx = _mm256_setzero_ps();
	float sum = 0.0f;
	std::vector<float> vec_aux(8);

	if (vec1.size() == vec2.size())
	{
		for (int i = 0; i < vec1.size(); i = i + 8)
		{
			vec1_avx = _mm256_loadu_ps(&vec1[i]);
			vec2_avx = _mm256_loadu_ps(&vec2[i]);
			res_avx = _mm256_fmadd_ps(vec1_avx, vec2_avx, res_avx);
		}
	}

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

//	computes the L2 norm of a vector
float norm(vector<float>& vec)
{
	float norm = 0.0f;

	for (int i = 0; i < vec.size(); i++)
		norm += vec[i] * vec[i];

	return (float)sqrt(norm);
}


