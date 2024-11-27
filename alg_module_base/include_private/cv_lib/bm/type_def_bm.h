#ifndef __CV_LIB_TYPE_DEF_BM_H__
#define __CV_LIB_TYPE_DEF_BM_H__
#include "cv_lib/type_def.h"

#include<bmruntime.h>
#include<bmruntime_cpp.h>
#include<bmcv_api.h>
#include<common.h>
#define USE_BM
struct Image_bm_
{																							//计算卡的图像数据
	bm_image image;
	bm_handle_t handle;
	~Image_bm_(){};
};


class QyImage_bm:public QyImage{
    std::map<bm_image_format_ext,bm_image_format_ext> src_type2dst_type;
    std::map<bm_image_format_ext,bm_image_format_ext> src_channel_exchange_dst_channel;

    std::map<bm_image_format_ext,bm_image_format_ext> chw2hwc_map;
    std::map<bm_image_format_ext,bm_image_format_ext> hwc2chw_map;

public:
    Image_bm_ data;
    QyImage_bm(std::shared_ptr<Device_Handle> handle):QyImage(handle){
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_BGR_PACKED,bm_image_format_ext::FORMAT_BGR_PLANAR));
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_RGB_PACKED,bm_image_format_ext::FORMAT_RGB_PLANAR));
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_NV12,bm_image_format_ext::FORMAT_BGR_PLANAR));
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_NV21,bm_image_format_ext::FORMAT_BGR_PLANAR));
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_YUV420P,bm_image_format_ext::FORMAT_BGR_PLANAR));
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_BGR_PLANAR,bm_image_format_ext::FORMAT_BGR_PLANAR));
        src_type2dst_type.insert(std::make_pair(bm_image_format_ext::FORMAT_RGB_PLANAR,bm_image_format_ext::FORMAT_RGB_PLANAR));

        src_channel_exchange_dst_channel.insert(std::make_pair(bm_image_format_ext::FORMAT_BGR_PACKED,bm_image_format_ext::FORMAT_RGB_PACKED));
        src_channel_exchange_dst_channel.insert(std::make_pair(bm_image_format_ext::FORMAT_RGB_PACKED,bm_image_format_ext::FORMAT_BGR_PACKED));
        src_channel_exchange_dst_channel.insert(std::make_pair(bm_image_format_ext::FORMAT_BGR_PLANAR,bm_image_format_ext::FORMAT_RGB_PLANAR));
        src_channel_exchange_dst_channel.insert(std::make_pair(bm_image_format_ext::FORMAT_RGB_PLANAR,bm_image_format_ext::FORMAT_BGR_PLANAR));

        chw2hwc_map.insert(std::make_pair(bm_image_format_ext::FORMAT_BGR_PLANAR,bm_image_format_ext::FORMAT_BGR_PACKED));
        chw2hwc_map.insert(std::make_pair(bm_image_format_ext::FORMAT_RGB_PLANAR,bm_image_format_ext::FORMAT_RGB_PACKED));

        hwc2chw_map.insert(std::make_pair(bm_image_format_ext::FORMAT_BGR_PACKED,bm_image_format_ext::FORMAT_BGR_PLANAR));
        hwc2chw_map.insert(std::make_pair(bm_image_format_ext::FORMAT_RGB_PACKED,bm_image_format_ext::FORMAT_RGB_PLANAR));

    };
    virtual ~QyImage_bm(){
        if(this->owned){
            if(bm_image_is_attached(this->data.image)){
                bm_image_destroy(this->data.image);
            }
        }
        else{
            if(bm_image_is_attached(this->data.image)){
                bm_image_detach(this->data.image);
            }
        }
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

    virtual cv::Mat get_image();
    virtual void set_image(cv::Mat input,bool is_rgb=false);

    virtual std::shared_ptr<QyImage> operator*(cv::Scalar value);
    virtual std::shared_ptr<QyImage> operator/(cv::Scalar value);
    virtual std::shared_ptr<QyImage> operator+(cv::Scalar value);
    virtual std::shared_ptr<QyImage> operator-(cv::Scalar value);
    virtual std::shared_ptr<QyImage> scale_add(cv::Scalar factor,cv::Scalar value);


    virtual std::shared_ptr<QyImage> to_HWC();
    virtual std::shared_ptr<QyImage> to_CHW();
    virtual std::shared_ptr<QyImage> auto_swap_HWC(Shape_t shape);
};

#endif