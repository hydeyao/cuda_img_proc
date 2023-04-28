#include "pch.h"
#include "CuImageHandle.h"
#include "cuda.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void waveletThreshold(float* d_data, float threshold, int width, int height) {
    // �����߳�����
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ��������Ƿ�Խ��
    if (x < width && y < height) {
        int index = y * width + x;

        // ��ÿ��С��ϵ��������ֵ����
        if (abs(d_data[index]) < threshold) {
            d_data[index] = 0;
        }
    }
}



using namespace std;

CuImageHandle::CuImageHandle(ST_IMG* img_param):_mspStImgParam(img_param)
{
    cudalib_init();
	_mImgWidth = _mspStImgParam->width;
	_mImgheight = _mspStImgParam->height;
	_mspfImgBuf.reset(_mspStImgParam->img_buf);

}

CuImageHandle::CuImageHandle(float* img_buf, unsigned int width, unsigned int height): \
_mImgWidth(width), _mspfImgBuf(img_buf), _mImgheight(height)
{
    cudalib_init();
	_mspStImgParam.reset(new ST_IMG(img_buf, width, height));
}

CuImageHandle::CuImageHandle(cv::Mat src_mat):_mSrcMat(src_mat)
{
    cudalib_init();
}

CuImageHandle::~CuImageHandle()
{
    cudalib_cleanup();
}

void CuImageHandle::cudalib_init()
{
	cudaSetDevice(0);
}

void CuImageHandle::cudalib_cleanup()
{
	cudaDeviceReset();
}

void CuImageHandle::split_channle()
{
    cv::split(_mSrcMat, mchannle_vec);
}

cv::Mat CuImageHandle::cu_get_denoise_img()
{
    split_channle();



    return cv::Mat();
}

void CuImageHandle::cu_img_denoise()
{
    if (_mspfImgBuf)
    {
        cu_img_denoise(_mspfImgBuf.get(), _mImgWidth, _mImgheight);
    }
}

void CuImageHandle::cu_img_denoise(float* d_data, int width, int height)
{
    // ����CUDA����
    float* d_workspace;
    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_R2C);
    cudaMalloc(&d_workspace, height * (width / 2 + 1) * sizeof(cufftComplex));

    // ִ�п��ٸ���Ҷ�任
    cufftExecR2C(plan, d_data, (cufftComplex*)d_data);

    // ��С��ϵ��������ֵ����
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    waveletThreshold << <numBlocks, threadsPerBlock >> > (d_data, 10.0f, width, height);

    // ִ������ٸ���Ҷ�任
    cufftExecC2R(plan, (cufftComplex*)d_data, d_data);

    // �ͷ�CUDA����
    cufftDestroy(plan);
    cudaFree(d_workspace);

}
