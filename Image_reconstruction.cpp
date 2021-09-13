#include <iostream>
#include <thread>
#include <vector>
#include "sparse_representations.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void recostruct(cv::Mat &img, cv::Mat& out, float p)
{

	int width = img.size[1];
	int height = img.size[0];
	int n = width * height;
	int m = n / p;// int(n * p);
	int T = 12;
	int k = 0;
	vector<float> ek(n);
	vector<float> psi(n);
	vector<float> x(n);
	vector<float> x1(n);
	vector<float> x_aux(n);
	vector<float> x_map(n);
	vector<float> y(m);
	vector<float> s1(n);
	vector<vector<float>> Theta(m, vector<float>(n));
	vector<vector<float>> Phi(m, vector<float>(n));
	vector<vector<float>> Phi_t(n, vector<float>(m));
	vector<vector<float>> Theta_t(n, vector<float>(m));

	std::random_device e;
	std::default_random_engine generator(e());
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	static std::uniform_real_distribution<> dis(0, n-1);
	vec_fill(x_map, 0.0f);
	mat_fill(Phi, 0.0f, T);

	cv::Mat floatImg;
	cv::Mat reconstructedImg;
	cv::Mat originalImg;
	img.convertTo(floatImg, CV_32FC1);
	img.convertTo(reconstructedImg, CV_32FC1);
	img.convertTo(originalImg, CV_32FC1);

	k = 0;

	for (int i = 0; i < floatImg.rows; i++)
	{
		for (int j = 0; j < floatImg.cols; j++)
		{
			x[k++] = floatImg.at<float>(j, i);
		}
	}

	sparse_vec_binary(x_map, n / 4);
	for (int i = 0; i < x_map.size(); i++)
	{
		x_map[i] = x_map[i] * x[i];
	}
	//x = x_map;
	//vec_print(x_map);
	/*
	#pragma omp parallel for num_threads(T) schedule(dynamic)
	for (int i = 0; i < Phi.size(); i++)
	{
		sparse_vec_binary(Phi[i], n / 2);
		vec_add_avx(x_map, Phi[i], 2);
	}*/
	mat_rand(Phi, T);

	x_aux = x;

	vector<float> res(m);
	int t = 32;
	int m1 = 32;
	float sum = 0.0f;

	
	mat_transpose(Phi, Phi_t, 16);

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
	vector<float> flat_mat(Phi_t.size() * Phi_t[0].size());
	flatten_scalar(Phi_t, flat_mat, 16);
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * flat_mat.size());
	cl::Buffer buffer_vec(context, CL_MEM_READ_WRITE, sizeof(float) * x_aux.size());
	cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(float) * res.size());
	cl::CommandQueue queue(context, default_device);
	cl::Kernel kernel_mat_vec_mul_gpu;
	kernel_mat_vec_mul_gpu = cl::Kernel(program, "mat_vec_mul_gpu_sp");
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * flat_mat.size(), flat_mat.data());
	queue.enqueueWriteBuffer(buffer_vec, CL_TRUE, 0, sizeof(float) * x_aux.size(), x_aux.data());
	queue.enqueueWriteBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
	kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
	kernel_mat_vec_mul_gpu.setArg(1, buffer_vec);
	kernel_mat_vec_mul_gpu.setArg(2, buffer_res);
	kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
	kernel_mat_vec_mul_gpu.setArg(4, m);
	kernel_mat_vec_mul_gpu.setArg(5, n);
	queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));
	queue.finish();
	queue.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
	queue.finish();
	x_aux = res;
	
	y = x_aux;

	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);
		
		queue.enqueueWriteBuffer(buffer_vec, CL_TRUE, 0, sizeof(float) * psi.size(), psi.data());
		kernel_mat_vec_mul_gpu.setArg(0, buffer_A);
		kernel_mat_vec_mul_gpu.setArg(1, buffer_vec);
		kernel_mat_vec_mul_gpu.setArg(2, buffer_res);
		kernel_mat_vec_mul_gpu.setArg(3, m1 * t * sizeof(float), NULL);
		kernel_mat_vec_mul_gpu.setArg(4, m);
		kernel_mat_vec_mul_gpu.setArg(5, n);
		queue.enqueueNDRangeKernel(kernel_mat_vec_mul_gpu, cl::NullRange, cl::NDRange(m, t), cl::NDRange(m1, t));
		queue.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
		queue.finish();
		psi = res;

		Theta_t[i] = psi;
	}

	mat_transpose(Theta_t, Theta, T);

	adm <float>* data1;
	data1 = new adm<float>(Theta, y, 0.000001f, 0.000001f, 1000);
	data1->set_number_of_threads(T);
	s1 = data1->solution_gpu();

	vec_fill(x1, 0.0f);
	vec_fill(ek, 0.0f);
	vec_fill(psi, 0.0f);

	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);
		vec_scalar_avx(psi, s1[i], T);
		vec_add_avx(x1, psi, T);
	}

	k = 0;

	for (int i = 0; i < floatImg.rows; i++)
	{
		for (int j = 0; j < floatImg.cols; j++)
		{
			reconstructedImg.at<float>(j, i) = x1[k++];
		}
	}
	k = 0;

	for (int i = 0; i < floatImg.rows; i++)
	{
		for (int j = 0; j < floatImg.cols; j++)
		{
			originalImg.at<float>(j, i) = x_map[k++];
		}
	}

	cv::Mat dst;
	reconstructedImg.convertTo(dst, CV_8U);
	cv::Mat org1;
	originalImg.convertTo(org1, CV_8U);


	out = dst;
}

int main()
{
	int k = 0;
	cv::Mat img = cv::imread("C:\\images\\lenna.jpg", cv::IMREAD_GRAYSCALE);

	if (img.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	float p = 2;

	cv::Mat channel_1;
	cv::Mat channel_2;
	cv::Mat channel_3;
	cv::Mat out_1;
	cv::Mat out_2;
	cv::Mat out_3;
	cv::Mat org_1;
	cv::Mat org_2;
	cv::Mat org_3;
	cv::Mat reconstructed_img;
	cv::Mat original_img;
	

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	recostruct(img, out_1, p);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	float solve_time = duration_cast<milliseconds>(t2 - t1).count();
	cout << solve_time << endl;

	cv::imwrite("C:/Users/iorda/Desktop/lenna_n_1.bmp", out_1);
	imshow("Display window", out_1);
	k = cv::waitKey(0);
	
	return 1;
}
