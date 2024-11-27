
#ifndef __CV_LIB_TYPE_DEF_H__
#define __CV_LIB_TYPE_DEF_H__

#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#include "network_engine/device_handle.h"
class QyImage{
public:
    enum Padding_mode
	{
        Center,                             //将原图像放到目标画布中心
        LeftTop                             //将原图像放到目标画布左上角
    };

	enum Data_type{
        Float64,                            //opencv
        Float32,                            //opencv,bm,cvcuda
        Int32,                              //opencv
        Float16,                            //opencv,bm,cvcuda
        UInt16,                             //opencv
        Int16,                              //opencv
        UInt8,                              //opencv,bm,cvcuda
    };

protected:

    bool owned=false;                                   //部分分支下，需要通过ownd判断对象是否拥有设备内存的指针
    bool is_rgb=false;                                  //is_rgb为false时图像为BGR，为true时图像为RGB
    std::shared_ptr<Device_Handle> handle;              //图像所在设备的Handle
public:    
    QyImage(std::shared_ptr<Device_Handle> handle);
    virtual ~QyImage();


    virtual void set_owned(bool owned);                 //设置是否拥有设备内存
    void set_is_rgb(bool is_rgb);                       //设置图像RGB模式
    bool get_is_rgb();                       //设置图像RGB模式

    virtual bool is_empty()=0;                          //图像是否为空，长宽为0或没有分配设备内存


    virtual int get_width()=0;                          //获取图像宽度
    virtual int get_height()=0;                         //获取图像高度

    virtual std::shared_ptr<QyImage> copy()=0;          //获取图像的深拷贝
    virtual std::shared_ptr<Device_Handle> get_handle();//获取图像的handle

    virtual std::shared_ptr<QyImage> crop(cv::Rect box)=0;      //获取剪裁区域内的图像的拷贝
    virtual std::shared_ptr<QyImage> resize(int width,int height,bool use_bilinear=true)=0;     //图像缩放到指定尺寸
    virtual std::shared_ptr<QyImage> crop_resize(cv::Rect box,int width,int height,bool use_bilinear=true)=0;   //剪裁图像并将剪裁后图像缩放到指定尺寸
    virtual std::shared_ptr<QyImage> resize_keep_ratio(int width,int height,int value,Padding_mode mode=Padding_mode::LeftTop,bool use_bilinear=true);  //保持原图长宽比的情况下，将图像缩放到指定尺寸，尺寸大于计算后原图缩放后尺寸的部分会进行padding
    virtual std::shared_ptr<QyImage> crop_resize_keep_ratio(cv::Rect box,int width,int height,int value,Padding_mode mode=Padding_mode::LeftTop,bool use_bilinear=true);    //剪裁图像，并在保持剪裁图像长宽比的情况下，将剪裁图像缩放到指定尺寸，并padding

    virtual std::shared_ptr<QyImage> padding(int left,int right,int up,int down,int value)=0;   //在图像左右上下padding
    virtual std::shared_ptr<QyImage> padding_to(int width,int height,int value,Padding_mode mode=Padding_mode::LeftTop);    //将图像padding到指定尺寸


    virtual std::shared_ptr<QyImage> warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear=false)=0;    //仿射变换
    virtual std::shared_ptr<QyImage> warp_affine(cv::Mat& matrix,int width,int height,bool use_bilinear=false);                 //仿射变换
    virtual std::vector<std::shared_ptr<QyImage>> batch_warp_affine(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false);  //批量仿射变换，部分硬件下，效率比循环进行单个仿射变换效率高
    virtual std::shared_ptr<QyImage> warp_perspective(cv::Mat& matrix,int width,int height,bool use_bilinear=false);            //透射变换

    virtual std::shared_ptr<QyImage> warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear=false)=0;            //透射变换
    virtual std::shared_ptr<QyImage> warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear=false)=0;            //透射变换
    virtual std::vector<std::shared_ptr<QyImage>> batch_warp_perspective(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false);            //批量透射变换，部分硬件下，效率比循环进行单个仿射变换效率高
    virtual std::vector<std::shared_ptr<QyImage>> batch_warp_perspective(std::vector<std::vector<cv::Point2f>>& pointss,int width,int height,bool use_bilinear=false);          //批量透射变换，部分硬件下，效率比循环进行单个仿射变换效率高

    virtual std::shared_ptr<QyImage> cvtcolor(bool to_rgb=false)=0;     //图像模式转为RGB或BGR

    virtual std::shared_ptr<QyImage> convertTo(Data_type t)=0;          //图像数据格式转换，uint8->float16等

    virtual cv::Mat get_image()=0;                                      //获取opencv格式的Mat

    virtual void set_image(cv::Mat input,bool is_rgb=false)=0;          //设置opencv格式的Mat


};
class InputOutput;

//从设备地址或内存地址构建QyImage，部分情况下可用此函数避免device和host之间的拷贝，直接从device到device构建QyImage对象
extern "C" std::shared_ptr<QyImage> from_data(uint8_t* img_data,int element_size,int src_stride,int width,int height,std::shared_ptr<Device_Handle> handle,bool is_rgb=false,bool from_host=false);
//从Opencv的Mat构建QyImage（常用）
extern "C" std::shared_ptr<QyImage> from_mat(cv::Mat img,std::shared_ptr<Device_Handle> handle,bool is_rgb=false);
//从InputOutput构建QyImage
extern "C" std::shared_ptr<QyImage> from_InputOutput(std::shared_ptr<InputOutput> src,bool copy=false);


#endif