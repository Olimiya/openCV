#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 定义总数据矩阵的行数和列数
#define ROWS 15000
#define COLS 30

// 定义每一块内的线程个数，GT720最多是1024（必须大于总矩阵的列数：30）
#define NUM_THREADS 1024


bool InitCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

__device__ float meanForRankCUDA(int num)
{
	float sum = 0;
	for (int i = 0; i <= num; i++) {
		sum += i;
	}
	return sum / (num + 1);
}


__device__ float meanForArrayCUDA(float array[], int len)
{
	float sum = 0;
	for (int i = 0; i < len; i++) {
		sum += array[i];
	}
	return sum / len;
}


__device__ float spearmanKernel(int Xarray[], int Yarray[])
{
	//1，对原先的数据进行排序，相同的值取平均值
	float Xrank[30];
	float Yrank[30];
	int col = 30;

	for (int i = 0; i < col; i++) {
		int bigger = 1;
		int equaer = -1;
		for (int j = 0; j < col; j++) {
			if (Xarray[i] < Xarray[j]) {
				bigger = bigger + 1;
			}
			else if (Xarray[i] == Xarray[j]) {
				equaer = equaer + 1;
			}
		}
		Xrank[i] = bigger + meanForRankCUDA(equaer);
	}
	for (int i = 0; i < col; i++) {
		int bigger = 1;
		int equaer = -1;
		for (int j = 0; j < col; j++) {
			if (Yarray[i] < Yarray[j]) {
				bigger = bigger + 1;
			}
			else if (Yarray[i] == Yarray[j]) {
				equaer = equaer + 1;
			}
		}
		Yrank[i] = bigger + meanForRankCUDA(equaer);
	}

	//2，计算斯皮尔曼相关性系数
	float numerator = 0;
	float denominatorLeft = 0;
	float denominatorRight = 0;
	float meanXrank = meanForArrayCUDA(Xrank, col);
	float meanYrank = meanForArrayCUDA(Yrank, col);
	for (int i = 0; i < col; i++) {
		numerator += (Xrank[i] - meanXrank) * (Yrank[i] - meanYrank);
		denominatorLeft += powf(Xrank[i] - meanXrank, 2);
		denominatorRight += powf(Yrank[i] - meanYrank, 2);
	}
	float corr = 0;
	if ((denominatorLeft != 0) && (denominatorRight != 0)) {
		corr = numerator / sqrtf(denominatorLeft * denominatorRight);
	}
	return corr;
}


__global__ static void spearCUDAShared(const int* a, size_t lda, float* c, size_t ldc, float* d, size_t ldd)
{
	extern __shared__ int data[];
	const int tid = threadIdx.x;
	const int row = blockIdx.x;
	int i, j;
	// 同步第1行~倒数第二行到共享内存，行数由block个数（总数据矩阵的行数-1）控制，每个block共享一行数据
	if (tid < 30) {
		data[tid] = a[row * lda + tid];
	}
	__syncthreads();

	int cal_per_block = gridDim.x - row; // 每个块分担的计算量
	int cal_per_thread = cal_per_block / blockDim.x + 1; // 每个线程分担的计算量
	// 分配各线程计算任务，通过for循环控制在一个线程需要计算的组数
	for (i = row + cal_per_thread * tid; i < (row + cal_per_thread * (tid + 1)) && i < gridDim.x; i++) {
		int j_row[30]; // 存放总数据矩阵的第j行
		for (j = 0; j < 30; j++) {
			j_row[j] = a[(i + 1)*lda + j];
		}
		float corr = spearmanKernel(data, j_row);
		c[row * ldc + (i + 1)] = corr;
		float t_test = 0;
		if (corr != 0) t_test = corr * (sqrtf((30 - 2) / (1 - powf(corr, 2))));
		d[row * ldd + (i + 1)] = t_test;
		//printf("block号：%d, 线程号：%d, 计算组：%d-%d, id号：%d, block个数：%d, 每块线程个数：%d, 该块总计算量：%d, 该块中每个线程计算量：%d, corr: %lf, %d, %d, %d - %d, %d, %d\n", row, tid, row, i + 1, (row*blockDim.x + tid), gridDim.x, blockDim.x, cal_per_block, cal_per_thread, corr, data[0], data[1], data[29], j_row[0], j_row[1], j_row[29]);
	}
}


clock_t matmultCUDA(const int* a, float* c, float* d)
{
	int *ac;
	float *cc, *dc;
	clock_t start, end;
	start = clock();

	size_t pitch_a, pitch_c, pitch_d;
	// 开辟a、c、d在GPU中的内存
	cudaMallocPitch((void**)&ac, &pitch_a, sizeof(int)* COLS, ROWS);
	cudaMallocPitch((void**)&cc, &pitch_c, sizeof(float)* ROWS, ROWS);
	cudaMallocPitch((void**)&dc, &pitch_d, sizeof(float)* ROWS, ROWS);
	// 复制a从CPU内存到GPU内存
	cudaMemcpy2D(ac, pitch_a, a, sizeof(int)* COLS, sizeof(int)* COLS, ROWS, cudaMemcpyHostToDevice);

	spearCUDAShared << <ROWS - 1, NUM_THREADS, sizeof(int)* COLS >> > (ac, pitch_a / sizeof(int), cc, pitch_c / sizeof(float), dc, pitch_d / sizeof(float));

	cudaMemcpy2D(c, sizeof(float)* ROWS, cc, pitch_c, sizeof(float)* ROWS, ROWS, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(d, sizeof(float)* ROWS, dc, pitch_d, sizeof(float)* ROWS, ROWS, cudaMemcpyDeviceToHost);
	cudaFree(ac);
	cudaFree(cc);

	end = clock();
	return end - start;
}


void print_int_matrix(int* a, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%d\t", a[i * col + j]);
		}
		printf("\n");
	}
}


void print_float_matrix(float* c, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%f\t", c[i * col + j]);
		}
		printf("\n");
	}
}

void read_ints(int* a) {
	FILE* file = fopen("D:\\MASTER2016\\5.CUDA\\data-ID-top30-kv.txt", "r");
	int i = 0;
	int count = 0;

	fscanf(file, "%d", &i);
	while (!feof(file))
	{
		a[count] = i;
		count++;
		if (count == ROWS * COLS) break;
		fscanf(file, "%d", &i);
	}
	fclose(file);
}


int main()
{
	int *a; // CPU内存中的总数据矩阵，ROWS行，COLS列
	float *c; // CPU内存中的相关系数结果矩阵，ROWS行，ROWS列
	float *d; // CPU内存中的T值结果矩阵，ROWS行，ROWS列
	a = (int*)malloc(sizeof(int)* COLS * ROWS);
	c = (float*)malloc(sizeof(float)* ROWS * ROWS);
	d = (float*)malloc(sizeof(float)* ROWS * ROWS);

	clock_t start = clock();
	printf(">> loading ... rows: %d, cols: %d", ROWS, COLS);
	read_ints(a);
	clock_t end = clock() - start;
	printf("\nTime used: %.2f s\n", (double)(end) / CLOCKS_PER_SEC);

	//print_int_matrix(a, ROWS, COLS);
	//printf("\n");

	printf(">> calculating ... ");
	printf("\n---------------------------------------");
	printf("\ntotal groups: %lld", (long long)ROWS*(ROWS - 1) / 2);
	printf("\ntotal threads: %d (blocks) * 1024 = %d", (ROWS - 1), (ROWS - 1) * 1024);
	printf("\ntotal space complexity: %lld MB", (long long)((ROWS / 1024) * (ROWS / 1024) * 8));
	printf("\n---------------------------------------");
	if (!InitCUDA()) return 0;
	clock_t time = matmultCUDA(a, c, d);
	double sec = (double)(time + end) / CLOCKS_PER_SEC;
	printf("\nTime used: %.2f s\n", sec);

	printf(">> saving ... ");
	FILE *f = fopen("D:\\MASTER2016\\5.CUDA\\result-c-2.txt", "w");
	for (int i = 0; i < ROWS; i++) {
		for (int j = i + 1; j < ROWS; j++) {
			float t_test = d[i * ROWS + j];
			if (t_test >= 2.042) {
				fprintf(f, "X%d\tX%d\t%f\t%lf\n", i + 1, j + 1, c[i * ROWS + j], t_test);
			}
		}
	}
	fclose(f);
	end = clock() - start;
	printf("OK\nTime used: %.2f s\n", (double)(end) / CLOCKS_PER_SEC);

	//printf(">> 相关系数结果矩阵: \n");
	//print_float_matrix(c, ROWS, ROWS);
	//printf(">> T值结果矩阵: \n");
	//print_float_matrix(d, ROWS, ROWS);

	getchar();
	return 0;
}