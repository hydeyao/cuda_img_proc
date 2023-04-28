#pragma once
#ifndef __CU_IMAGE_HANDLE_H__
#define __CU_IMAGE_HANDLE_H__

#define CU_IMG_HANDLE_EXPORT __declspec(dllexport)

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>


struct ST_IMG
{
	float* img_buf = nullptr;
	unsigned int width = 0;
	unsigned int height = 0;
	ST_IMG(float* _img_buf, unsigned int _width, unsigned int _height)
	{
		_img_buf = img_buf;
		_width = width;
		_height = height;
	};
};

class CU_IMG_HANDLE_EXPORT CuImageHandle
{
public:
	CuImageHandle(ST_IMG* img_param);
	CuImageHandle(float* img_buf, unsigned int width, unsigned int height);
	CuImageHandle(cv::Mat src_mat);
	~CuImageHandle();

private:
	std::shared_ptr<ST_IMG> _mspStImgParam = nullptr;
	std::shared_ptr<float> _mspfImgBuf = nullptr;
	unsigned int _mImgWidth = 0;
	unsigned int _mImgheight = 0;

	float* md_data[3];

	cv::Mat _mSrcMat;
	std::vector<cv::Mat> mchannle_vec;
	

public:
	cv::Mat cu_get_denoise_img();
	


private:
	void cu_img_denoise();
	void cu_img_denoise(float* d_data, int width, int height);
	void cudalib_init();
	void cudalib_cleanup();

	void split_channle();

};







#endif // !__CU_IMAGE_HANDLE_H__




