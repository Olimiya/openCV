#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <opencv2\opencv_modules.hpp>
#include <iostream>
#include <time.h>
using namespace std;

// 定义测试矩阵的维度
int const M = 5;
int const N = 10;

extern "C"
void addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C"
void getDeviceInfo();
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
int MatMultiply();

//int main()
//{
//    getDeviceInfo();
//    MatMultiply();
//    std::cout << std::endl << std::endl;

//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };

//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }

//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);

//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//    return 0;
//}

extern "C"
void getDeviceInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        int driver_version(0), runtime_version(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0)
            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
                printf("\n");
        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
        printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total amount of Global Memory:                  %zu bytes\n", deviceProp.totalGlobalMem);
        printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
        printf("Total amount of Constant Memory:                %zu bytes\n", deviceProp.totalConstMem);
        printf("Total amount of Shared Memory per block:        %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
        printf("Warp size:                                      %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch:                           %zu bytes\n", deviceProp.memPitch);
        printf("Texture alignmemt:                              %zu bytes\n", deviceProp.texturePitchAlignment);
        printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
    }

}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

}

// 矩阵乘法
int MatMultiply()
{
    // 定义状态变量
    cublasStatus_t status;

    // 在 内存 中为将要计算的矩阵开辟空间
    float *h_A = (float*)malloc(N*M * sizeof(float));
    float *h_B = (float*)malloc(N*M * sizeof(float));

    // 在 内存 中为将要存放运算结果的矩阵开辟空间
    float *h_C = (float*)malloc(M*M * sizeof(float));

    srand((unsigned int)time(0));
    // 为待运算矩阵的元素赋予 0-10 范围内的随机数
    for (int i = 0; i < N*M; i++) {
        h_A[i] = (float)(rand() % 10 + 1);
        h_B[i] = (float)(rand() % 10 + 1);

    }

    // 打印待测试的矩阵
    std::cout << "矩阵 A :" << std::endl;
    for (int i = 0; i < N*M; i++) {
        cout << h_A[i] << " ";
        if ((i + 1) % N == 0) cout << endl;
    }
    cout << endl;
    cout << "矩阵 B :" << endl;
    for (int i = 0; i < N*M; i++) {
        cout << h_B[i] << " ";
        if ((i + 1) % M == 0) cout << endl;
    }
    cout << endl;

    /*
    ** GPU 计算矩阵相乘
    */

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS 对象实例化出错" << endl;
        }
        getchar();
        return EXIT_FAILURE;
    }

    float *d_A, *d_B, *d_C;
    // 在 显存 中为将要计算的矩阵开辟空间
    cudaMalloc(
        (void**)&d_A,    // 指向开辟的空间的指针
        N*M * sizeof(float)    //　需要开辟空间的字节数
    );
    cudaMalloc(
        (void**)&d_B,
        N*M * sizeof(float)
    );

    // 在 显存 中为将要存放运算结果的矩阵开辟空间
    cudaMalloc(
        (void**)&d_C,
        M*M * sizeof(float)
    );

    // 将矩阵数据传递进 显存 中已经开辟好了的空间
    cublasSetVector(
        N*M,    // 要存入显存的元素个数
        sizeof(float),    // 每个元素大小
        h_A,    // 主机端起始地址
        1,    // 连续元素之间的存储间隔
        d_A,    // GPU 端起始地址
        1    // 连续元素之间的存储间隔
    );
    cublasSetVector(
        N*M,
        sizeof(float),
        h_B,
        1,
        d_B,
        1
    );

    // 同步函数
    cudaThreadSynchronize();

    // 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
    float a = 1; float b = 0;
    // 矩阵相乘。该函数必然将数组解析成列优先数组
    cublasSgemm(
        handle,    // blas 库对象
        CUBLAS_OP_T,    // 矩阵 A 属性参数
        CUBLAS_OP_T,    // 矩阵 B 属性参数
        M,    // A, C 的行数
        M,    // B, C 的列数
        N,    // A 的列数和 B 的行数
        &a,    // 运算式的 α 值
        d_A,    // A 在显存中的地址
        N,    // lda
        d_B,    // B 在显存中的地址
        M,    // ldb
        &b,    // 运算式的 β 值
        d_C,    // C 在显存中的地址(结果矩阵)
        M    // ldc
    );

    // 同步函数
    cudaThreadSynchronize();

    // 从 显存 中取出运算结果至 内存中去
    cublasGetVector(
        M*M,    //  要取出元素的个数
        sizeof(float),    // 每个元素大小
        d_C,    // GPU 端起始地址
        1,    // 连续元素之间的存储间隔
        h_C,    // 主机端起始地址
        1    // 连续元素之间的存储间隔
    );

    // 打印运算结果
    cout << "计算结果的转置 ( (A*B)的转置 )：" << endl;

    for (int i = 0; i < M*M; i++) {
        cout << h_C[i] << " ";
        if ((i + 1) % M == 0) cout << endl;
    }

    // 清理掉使用过的内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放 CUBLAS 库对象
    cublasDestroy(handle);

//    getchar();
    return 1;
}
