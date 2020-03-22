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


__global__ void BicubicScale(int *In, int *Out, int Width, int Height, int dstWidth, float xRatio, float yRatio)
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
        Out[iy* dstWidth + ix] = In[fy * Width + fx];
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
#define FILLPX(_x, _y, _i, _j) p[_i][_j]=In[(_y) * Width + (_x)]
        FILLPX(fx - 1, fy - 1, 0, 0);
        FILLPX(fx - 1, fy + 0, 0, 1);
        FILLPX(fx - 1, fy + 1, 0, 2);
        FILLPX(fx - 1, fy + 2, 0, 3);
        FILLPX(fx + 0, fy - 1, 1, 0);
        FILLPX(fx + 0, fy + 0, 1, 1);
        FILLPX(fx + 0, fy + 1, 1, 2);
        FILLPX(fx + 0, fy + 2, 1, 3);
        FILLPX(fx + 1, fy - 1, 2, 0);
        FILLPX(fx + 1, fy + 0, 2, 1);
        FILLPX(fx + 1, fy + 1, 2, 2);
        FILLPX(fx + 1, fy + 2, 2, 3);
        FILLPX(fx + 2, fy - 1, 3, 0);
        FILLPX(fx + 2, fy + 0, 3, 1);
        FILLPX(fx + 2, fy + 1, 3, 2);
        FILLPX(fx + 2, fy + 2, 3, 3);
#undef FILLPX
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
    Out[iy* dstWidth + ix] = rgb;
//    Out[iy* dstWidth + ix] = 100;
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
    BicubicScale << <dimGrid, dimBlock >> > (pixelIn, pixelOut, Width, Height, DstWidth, xRatio, yRatio);

//    checkCudaErrors(cudaMemcpy(out, pixelOut, sizeof(int) * Width * Height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(out, pixelOut, sizeof(int) * DstWidth * DstHeight, cudaMemcpyDeviceToHost));

    std::cout.flush();
	cudaFree(pixelIn);
	cudaFree(pixelOut);
}

#endif // ! __BicubicScale_CU_
