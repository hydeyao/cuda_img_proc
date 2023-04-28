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

    // ����CUDA�ڴ�
    unsigned char* dev_src;
    unsigned char* dev_dst;
    cudaMalloc((void**)&dev_src, size * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&dev_dst, size * sizeof(unsigned char));

    // ��ԭʼRGBͼ�����ݸ��Ƶ�CUDA�ڴ�
    cudaMemcpy(dev_src, src, size * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // ����CUDA�ں˺����Ŀ��С���̴߳�С
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // ����CUDA�ں˺�������ͼ����
    cuda_gray << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width, height);

    // ���Ҷ�ͼ�����ݴ�CUDA�ڴ渴�Ƶ������ڴ�
    cudaMemcpy(dst, dev_dst, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // �ͷ�CUDA�ڴ�
    cudaFree(dev_src);
    cudaFree(dev_dst);
}
