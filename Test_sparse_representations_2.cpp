#include <iostream>
#include <thread>
#include <vector>
#include "sparse_representations.h"

int main()
{
	int n = 256, m = 128;
	double time_cpu, time_gpu;
	vector<double> x1;
	vector<double> x2;
	vector<double> x3;

	vector<vector<double>> A(m, vector<double>(n));
	vector<double> b(m);
	vector<double> sol(n);
	vector<double> aux(n);
        
        // generating random data
	mat_rand(A, 16);
	// generating a random sparse vector with 5% non zero elements
	sparse_vec(sol, sol.size() * 0.05);
	aux = sol;
	mat_vec_mul_avx(A, aux, 16);
	b = aux;

	adm <double>* data1;
	data1 = new adm<double>(A, b, 0.001f, 0.001f, 1000);
	data1->set_number_of_threads(16);
	x1 = data1->solution_gpu();
	vec_sub_avx(x1, sol, 8);
	cout << "ADM error : " << norm(x1) << endl << "ADM solve time : " << data1->solve_time << endl;
	delete data1;

	fista <double>* data2;
	data2 = new fista<double>(A, b, 0.3e-2, 1000);;
	data2->set_number_of_threads(16);
	x2 = data2->solution_gpu();
	vec_sub_avx(x2, sol, 8);
	cout << "FISTA error : " << norm(x2) << endl << "FISTA solve time : " << data2->solve_time << endl;
	delete data2;

	palm<double>* data3;
	data3 = new palm<double>(A, b, 500, 3);
	data3->set_number_of_threads(16);
	x3 = data3->solution_gpu();
	vec_sub_avx(x3, sol, 8);
	cout << "PALM error : " << norm(x3) << endl << "PALM solve time : " << data3->solve_time << endl;
	delete data3;

	return 1;
}