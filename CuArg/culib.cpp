#include "pch.h"
#include "culib.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void cuda_gray(unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = y * width + x;
        unsigned char r = src[i * 3];
        unsigned char g = src[i * 3 + 1];
        unsigned char b = src[i * 3 + 2];
        dst[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}


void cudalib_init()
{
	cudaSetDevice(0);
}

void cudalib_cleanup()
{
	cudaDeviceReset();
}

void cudalib_process(unsigned char* src, unsigned char* dst, int width, int height)
{
    int size = width * height;

    // 分配CUDA内存
    unsigned char* dev_src;
    unsigned char* dev_dst;
    cudaMalloc((void**)&dev_src, size * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&dev_dst, size * sizeof(unsigned char));

    // 将原始RGB图像数据复制到CUDA内存
    cudaMemcpy(dev_src, src, size * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 计算CUDA内核函数的块大小和线程大小
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用CUDA内核函数进行图像处理
    cuda_gray << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width, height);

    // 将灰度图像数据从CUDA内存复制到主机内存
    cudaMemcpy(dst, dev_dst, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // 释放CUDA内存
    cudaFree(dev_src);
    cudaFree(dev_dst);
}
