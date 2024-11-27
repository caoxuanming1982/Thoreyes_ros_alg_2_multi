#ifndef __CV_LIB_TYPE_DEF_CUDA_H__
#define __CV_LIB_TYPE_DEF_CUDA_H__
#include "cv_lib/type_def.h"

#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#include<opencv2/cvconfig.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#define USE_CUDA

struct Image_cv_cuda_
{							
	int device_idx=0;																//计算卡的图像数据
	cv::cuda::GpuMat image;
	~Image_cv_cuda_(){};
};


class QyImage_cv_cuda:public QyImage{

public:
    Image_cv_cuda_ data;
    QyImage_cv_cuda(std::shared_ptr<Device_Handle> handle):QyImage(handle){
                
    }
    virtual ~QyImage_cv_cuda(){
    };
    virtual int get_width();
    virtual int get_height();
    virtual bool is_empty();

    virtual std::shared_ptr<QyImage> copy();
    virtual std::shared_ptr<QyImage> crop(cv::Rect box);
    virtual std::shared_ptr<QyImage> resize(int width,int height,bool use_bilinear=true);
    virtual std::shared_ptr<QyImage> crop_resize(cv::Rect box,int width,int height,bool use_bilinear=true);

    virtual std::shared_ptr<QyImage> padding(int left,int right,int up,int down,int value);

    virtual std::shared_ptr<QyImage> warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear=false);

    virtual std::shared_ptr<QyImage> warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear=false);
    virtual std::shared_ptr<QyImage> warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear=false);

    virtual std::shared_ptr<QyImage> cvtcolor(bool to_rgb=false);

    virtual std::shared_ptr<QyImage> convertTo(QyImage::Data_type t);

    virtual std::shared_ptr<QyImage> operator*(cv::Scalar value);
    virtual std::shared_ptr<QyImage> operator/(cv::Scalar value);
    virtual std::shared_ptr<QyImage> operator+(cv::Scalar value);
    virtual std::shared_ptr<QyImage> operator-(cv::Scalar value);
    virtual std::shared_ptr<QyImage> scale_add(cv::Scalar factor,cv::Scalar value);

    virtual cv::Mat get_image();
    virtual void set_image(cv::Mat input,bool is_rgb=false);


};

#endif