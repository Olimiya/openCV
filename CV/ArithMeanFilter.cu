﻿#ifndef  __MEDIANFILTER_CU_
#define  __MEDIANFILTER_CU_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#define datasize 100
extern "C" void ArithMedianFilter_host(int *pixel, int Width, int Height);

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
		return;
	}
}

__global__ void ArithMedianFilter(int *In, int *Out, int Width, int Height)
{
	int window[9];
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x <= Width && x >= 0 && y <= Height && y >= 0)
	{
//        if(x == 0 || y == 0 || x == Width -1 || y == Height - 1)
//        {
//            Out[y* Width + x] = In[y* Width + x];
//            return;
//        }
        window[0] = (y == 0 || x == 0) ? 125 : In[(y - 1)* Width + x - 1];
        window[1] = (y == 0) ? 125 : In[(y - 1)* Width + x];
        window[2] = (y == 0 || x == Width - 1) ? 125 : In[(y - 1)* Width + x + 1];
        window[3] = (x == 0) ? 125 : In[y* Width + x - 1];
		window[4] = In[y* Width + x];
        window[5] = (x == Width - 1) ? 125 : In[y* Width + x + 1];
        window[6] = (y == Height - 1 || x == 0) ? 125 : In[(y + 1)* Width + x - 1];
        window[7] = (y == Height - 1) ? 125 : In[(y + 1)* Width + x];
        window[8] = (y == Height - 1 || x == Width - 1) ? 125 : In[(y + 1)* Width + x + 1];
		int pixel = 0;
		for (int i = 0; i < 9; i++)
			pixel += window[i];
		pixel /= 9;
		//for (unsigned int j = 0; j < 5; j++)
		//{
		//	int min = j;
		//	for (unsigned int l = j + 1; l < 9; l++)
		//		if (window[l] < window[min])
		//			min = l;
		//	const float temp = window[j];
		//	window[j] = window[min];
		//	window[min] = temp;
		//}
		Out[y* Width + x] = pixel;
	}
}

extern "C" void ArithMedianFilter_host(int *pixel, int Width, int Height)
{
	int *pixelIn, *pixelOut;
	dim3 dimBlock(32, 32);
	dim3 dimGrid((Width + dimBlock.x - 1) / dimBlock.x, (Height + dimBlock.y -
		1) / dimBlock.y);
	checkCudaErrors(cudaMalloc((void**)&pixelIn, sizeof(int) * Width * Height));
	checkCudaErrors(cudaMalloc((void**)&pixelOut, sizeof(int) * Width * Height));

	checkCudaErrors(cudaMemcpy(pixelIn, pixel, sizeof(int) * Width * Height, cudaMemcpyHostToDevice));

	ArithMedianFilter << <dimGrid, dimBlock >> > (pixelIn, pixelOut, Width, Height);

	checkCudaErrors(cudaMemcpy(pixel, pixelOut, sizeof(int) * Width * Height, cudaMemcpyDeviceToHost));


	cudaFree(pixelIn);
	cudaFree(pixelOut);
}

#endif // ! __MEDIANFILTER_KERNEL_CU_
