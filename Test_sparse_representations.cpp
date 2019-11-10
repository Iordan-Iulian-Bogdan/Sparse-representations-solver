#include <iostream>
#include <vector>
#include "sparse_representations.h"

int main()
{
	//	This is a test program 

	string A_name = "A_test.bin";
	string b_name = "b_test.bin";
	vector<float>x1;
	vector<float>x2;
	vector<float>x3;

	adm<float>* data1;
	data1 = new adm<float>(A_name, b_name, 0.001f, 0.0001f, 1000);
	fista<float>* data2;
	data2 = new fista<float>(A_name, b_name, 0.3e-2, 1000);
	palm<float>* data3;
	data3 = new palm<float>(A_name, b_name, 1000, 3);

	//	compute the solutions with the CPU solvers  
	x1 = data1->solution_cpu();
	x2 = data2->solution_cpu();
	x3 = data3->solution_cpu();

	//	displays the erros and execution times for each algorithm
	cout << "Error for ADM (cpu): " << data1->error << '\n';
	cout << "Error for FISTA (cpu): " << data2->error << '\n';
	cout << "Error for PALM (cpu): " << data3->error << '\n';
	cout << '\n';

	cout << "ADM execution time (cpu): " << data1->solve_time / 1000 << " seconds" << '\n';
	cout << "FISTA execution time (cpu): " << data2->solve_time / 1000 << " seconds" << '\n';
	cout << "PALM execution time (cpu): " << data3->solve_time / 1000 << " seconds" << '\n';
	cout << '\n';

	adm<float>* data4;
	data4 = new adm<float>(A_name, b_name, 0.001f, 0.0001f, 1000);
	fista<float>* data5;
	data5 = new fista<float>(A_name, b_name, 0.3e-2, 1000);
	palm<float>* data6;
	data6 = new palm<float>(A_name, b_name, 1000, 3);

	//	compute the solutions for the same systems with the GPU solvers  
	x1 = data4->solution_gpu();
	x2 = data5->solution_gpu();
	x3 = data6->solution_gpu();

	cout << "Error for ADM (gpu): " << data4->error << '\n';
	cout << "Error for FISTA (gpu): " << data5->error << '\n';
	cout << "Error for PALM (gpu): " << data6->error << '\n';
	cout << '\n';

	cout << "ADM execution time (gpu): " << data4->solve_time / 1000 << " seconds, " << data1->solve_time / data4->solve_time << "x speedup" << '\n';
	cout << "FISTA execution time (gpu): " << data5->solve_time / 1000 << " seconds " << data2->solve_time / data5->solve_time << "x speedup" << '\n';
	cout << "PALM execution time (gpu): " << data6->solve_time / 1000 << " seconds " << data3->solve_time / data6->solve_time << "x speedup" << '\n';
	cout << '\n';

	vec_print(x3);

	delete data1;
	delete data2;
	delete data3;
	delete data4;
	delete data5;
	delete data6;

	return 1;
}
