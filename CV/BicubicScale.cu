#ifndef  __BicubicScale_CU_
#define  __BicubicScale_CU_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#define datasize 100
extern "C" void BicubicScale_host(int *in, int *out, int Width, int Height, int DstWidth, int DstHeight);

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
		return;
	}
}

__device__ double BicubicWeight(double x)
{
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return (a + 2.0)*x*x*x - (a + 3.0)*x*x + 1.0;
    else if (x < 2.0)
        return a * x*x*x - 5.0*a * x*x + 8.0*a * x - 4.0 * a;
    return 0.0;
}


__global__ void BicubicScale(int *In, int *Out, int Width, int Height, float xRatio, float yRatio)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x; //目标位置
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
    float x = ix / xRatio; //原始位置
    float y = iy / yRatio;
    int fx = (int)x; //原始位置
    int	fy = (int)y;

    // Handle the border
    if (fx - 1 <= 0 || fx + 2 >= Width - 1 || fy - 1 <= 0 || fy + 2 >= Height - 1)
    {
        fx = fx < 0 ? 0 : fx;
        fx = fx >= Width ? Width - 1 : fx;
        fy = fy < 0 ? 0 : fy;
        fy = fy >= Height ? Height - 1 : fy;
        Out[iy* Width + ix] = In[fy * Width + fx];
        return;
    }

    // Calc w
    double wx[4], wy[4];
    wx[0] = BicubicWeight(fx - 1 - x);
    wx[1] = BicubicWeight(fx + 0 - x);
    wx[2] = BicubicWeight(fx + 1 - x);
    wx[3] = BicubicWeight(fx + 2 - x);
    wy[0] = BicubicWeight(fy - 1 - y);
    wy[1] = BicubicWeight(fy + 0 - y);
    wy[2] = BicubicWeight(fy + 1 - y);
    wy[3] = BicubicWeight(fy + 2 - y);

    // Get pixels
    int p[4][4];
    p[0][0] = In[(fy - 1) * Width + fx - 1];
    p[0][1] = In[(fy - 1) * Width + fx - 1];
    p[0][2] = In[(fy - 1) * Width + fx-1];
    p[0][3] = In[(fy - 1) * Width + fx-1];

    p[1][0] = In[fy * Width + fx];
    p[1][1] = In[fy * Width + fx];
    p[1][2] = In[fy * Width + fx];
    p[1][3] = In[fy * Width + fx];

    p[2][0] = In[(fy + 1) * Width + fx+1];
    p[2][1] = In[(fy + 1) * Width + fx+1];
    p[2][2] = In[(fy + 1) * Width + fx+1];
    p[2][3] = In[(fy + 1) * Width + fx+1];

    p[3][0] = In[(fy + 1) * Width + fx+2];
    p[3][1] = In[(fy + 2) * Width + fx+2];
    p[3][2] = In[(fy + 2) * Width + fx+2];
    p[3][3] = In[(fy + 2) * Width + fx+2];

    double rgb = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
//            rgb += 125 * wx[i] * wy[j];
            rgb += p[i][j] * wx[i] * wy[j];
        }
    if (rgb < 0)
        rgb = 0;
    else if (rgb > 255)
        rgb = 255;
    Out[iy* Width + ix] = rgb;
}

extern "C" void BicubicScale_host(int *in, int *out, int Width, int Height, int DstWidth, int DstHeight)
{
	int *pixelIn, *pixelOut;
	dim3 dimBlock(32, 32);
	dim3 dimGrid((DstWidth + dimBlock.x - 1) / dimBlock.x, (DstHeight + dimBlock.y -
		1) / dimBlock.y);
	checkCudaErrors(cudaMalloc((void**)&pixelIn, sizeof(int) * Width * Height));
//    checkCudaErrors(cudaMalloc((void**)&pixelOut, sizeof(int) * Width * Height));
    checkCudaErrors(cudaMalloc((void**)&pixelOut, sizeof(int) * DstWidth * DstHeight));

	checkCudaErrors(cudaMemcpy(pixelIn, in, sizeof(int) * Width * Height, cudaMemcpyHostToDevice));
	
    float xRatio = (float)DstWidth / Width;
    float yRatio = (float)DstHeight / Height;
	BicubicScale << <dimGrid, dimBlock >> > (pixelIn, pixelOut, Width, Height, xRatio, yRatio);

//    checkCudaErrors(cudaMemcpy(out, pixelOut, sizeof(int) * Width * Height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(out, pixelOut, sizeof(int) * DstWidth * DstHeight, cudaMemcpyDeviceToHost));

    std::cout.flush();
	cudaFree(pixelIn);
	cudaFree(pixelOut);
}

#endif // ! __BicubicScale_CU_
