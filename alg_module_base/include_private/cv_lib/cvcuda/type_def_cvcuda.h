#ifndef __CV_LIB_TYPE_DEF_CVCUDA_H__
#include "cv_lib/type_def.h"

#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpResize.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpCopyMakeBorder.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/Rect.h>
#define USE_CVCUDA

struct Image_cvcuda_
{																							//计算卡的图像数据
    int device_idx=0;
	std::shared_ptr<nvcv::Tensor> image;
    Image_cvcuda_(){};
	~Image_cvcuda_(){};
};


class QyImage_cvcuda:public QyImage{

public:
    Image_cvcuda_ data;
    QyImage_cvcuda(std::shared_ptr<Device_Handle> handle):QyImage(handle){
                
    };
    virtual ~QyImage_cvcuda(){
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