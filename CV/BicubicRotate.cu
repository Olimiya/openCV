#ifndef  __BicubicScale_CU_
#define  __BicubicScale_CU_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#define datasize 100
extern "C" void BicubicRotate_host(int *in, int *out, int Width, int Height, int DstWidth,
	int DstHeight, float angle);

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
		return;
	}
}

__device__ double BicubicHermite(double A, double B, double C, double D, double t)
{
	double a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0;
	double b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0;
	double c = -A / 2.0 + C / 2.0;
	double d = B;
	return a * t * t * t + b * t * t + c * t + d;
}

__global__ void BicubicRotate(int *In, int *Out, int Width, int Height, int dstWidth, int dstHeight, 
	float angle)
{
    const double sina = sin(angle), cosa = cos(angle);
    dim3 ncenter = { dstWidth / 2.0, dstHeight / 2.0 }; //目标中心
    dim3 ocenter = { Width / 2.0, Height / 2.0 }; //原始中心

	int x = blockDim.x * blockIdx.x + threadIdx.x; //目标位置
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int xx = static_cast<int>(x - ncenter.x); 
	int yy = static_cast<int>(y - ncenter.y);
	double oldx = xx * cosa - yy * sina + ocenter.x; //原始位置
	double oldy = xx * sina + yy * cosa + ocenter.y;
	int iox = (int)oldx, ioy = (int)oldy; //原始位置
    Out[y* dstWidth + x] = 0;

    // out of interpolation border
    if (iox <= 1 || iox + 2 >= Width - 1 || ioy <= 1 || ioy + 2 >= Height)
    {
        // but, still in the original image
        if (iox >= 0 && iox < Width && ioy >= 0 && ioy < Height)
            Out[y * dstWidth + x] = In[ioy * Width + iox];
        return;
    }

    // Bicubic interpolation
    // 1st row
    auto p00 = In[(ioy - 1) * Width + (iox - 1)];
    auto p10 = In[(ioy - 1) * Width + (iox + 0)];
    auto p20 = In[(ioy - 1) * Width + (iox + 1)];
    auto p30 = In[(ioy - 1) * Width + (iox + 2)];

    // 2nd row
    auto p01 = In[(ioy + 0) * Width + (iox - 1)];
    auto p11 = In[(ioy + 0) * Width + (iox + 0)];
    auto p21 = In[(ioy + 0) * Width + (iox + 1)];
    auto p31 = In[(ioy + 0) * Width + (iox + 2)];

    // 3rd row
    auto p02 = In[(ioy + 1) * Width + (iox - 1)];
    auto p12 = In[(ioy + 1) * Width + (iox + 0)];
    auto p22 = In[(ioy + 1) * Width + (iox + 1)];
    auto p32 = In[(ioy + 1) * Width + (iox + 2)];

    // 4th row
    auto p03 = In[(ioy + 2) * Width + (iox - 1)];
    auto p13 = In[(ioy + 2) * Width + (iox + 0)];
    auto p23 = In[(ioy + 2) * Width + (iox + 1)];
    auto p33 = In[(ioy + 2) * Width + (iox + 2)];

    double result = 0;
    {
        double col0 = BicubicHermite(p00, p10, p20, p30, oldx - iox);
        double col1 = BicubicHermite(p01, p11, p21, p31, oldx - iox);
        double col2 = BicubicHermite(p02, p12, p22, p32, oldx - iox);
        double col3 = BicubicHermite(p03, p13, p23, p33, oldx - iox);
        result = BicubicHermite(col0, col1, col2, col3, oldy - ioy);
        if (result > 255.0)
            result = 255.0;
        if (result < 0.0)
            result = 0.0;
    }
    Out[y * dstWidth + x] = result;
}

extern "C" void BicubicRotate_host(int *in, int *out, int Width, int Height, int DstWidth, 
	int DstHeight, float angle)
{
	int *pixelIn, *pixelOut;
	dim3 dimBlock(32, 32);
	dim3 dimGrid((DstWidth + dimBlock.x - 1) / dimBlock.x, (DstHeight + dimBlock.y -
		1) / dimBlock.y);
	checkCudaErrors(cudaMalloc((void**)&pixelIn, sizeof(int) * Width * Height));
	//    checkCudaErrors(cudaMalloc((void**)&pixelOut, sizeof(int) * Width * Height));
	checkCudaErrors(cudaMalloc((void**)&pixelOut, sizeof(int) * DstWidth * DstHeight));

	checkCudaErrors(cudaMemcpy(pixelIn, in, sizeof(int) * Width * Height, cudaMemcpyHostToDevice));

	BicubicRotate << <dimGrid, dimBlock >> > (pixelIn, pixelOut, Width, Height, DstWidth, DstHeight,
		angle);

	//    checkCudaErrors(cudaMemcpy(out, pixelOut, sizeof(int) * Width * Height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(out, pixelOut, sizeof(int) * DstWidth * DstHeight, cudaMemcpyDeviceToHost));

	std::cout.flush();
	cudaFree(pixelIn);
	cudaFree(pixelOut);
}

#endif // ! __BicubicScale_CU_
