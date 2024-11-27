#include "cv_lib/cuda/type_dev_cuda.h"
#include "inout_type.h"
#include "network_engine/torch/device_handle_torch.h"
#include <cuda_runtime_api.h>

int QyImage_cv_cuda::get_width(){
    return this->data.image.cols;
};
int QyImage_cv_cuda::get_height(){
    return this->data.image.rows;
};
bool QyImage_cv_cuda::is_empty(){
    if(data.image.cols<=0||data.image.rows<=0)
        return true;
    return false;
};


std::shared_ptr<QyImage> QyImage_cv_cuda::copy(){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    result->data.image=this->data.image.clone();
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_cv_cuda::crop(cv::Rect box){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    result->data.image=this->data.image(box).clone();
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv_cuda::resize(int width,int height,bool use_bilinear){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    if(use_bilinear){
        cv::cuda::resize(this->data.image,result->data.image,cv::Size(width,height),0,0,cv::INTER_LINEAR);
    }
    else{
        cv::cuda::resize(this->data.image,result->data.image,cv::Size(width,height),0,0,cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv_cuda::crop_resize(cv::Rect box,int width,int height,bool use_bilinear){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::GpuMat temp=this->data.image(box);
    if(use_bilinear){
        cv::cuda::resize(temp,result->data.image,cv::Size(width,height),0,0,cv::INTER_LINEAR);
    }
    else{
        cv::cuda::resize(temp,result->data.image,cv::Size(width,height),0,0,cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cv_cuda::padding(int left,int right,int up,int down,int value){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::copyMakeBorder(this->data.image,result->data.image,up,down,left,right,cv::BORDER_CONSTANT,cv::Scalar(value,value,value));
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cv_cuda::warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::Mat M(2,3,CV_32FC1);
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            M.at<float>(i,j)=matrix[i*3+j];            
        }
    }
    if(use_bilinear){
        cv::cuda::warpAffine(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_LINEAR,cv::BORDER_REPLICATE);

    }
    else{
        cv::cuda::warpAffine(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_NEAREST,cv::BORDER_REPLICATE);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

std::shared_ptr<QyImage> QyImage_cv_cuda::warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::Mat M(3,3,CV_32FC1);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            M.at<float>(i,j)=matrix[i*3+j];            
        }
    }
    if(use_bilinear){
        cv::cuda::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::cuda::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv_cuda::warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear){
    cv::cuda::setDevice(this->data.device_idx);
    std::vector<cv::Point2f> coordinate_dst(4);
    coordinate_dst[0].x=0;
    coordinate_dst[0].y=0;
    coordinate_dst[1].x=width-1;
    coordinate_dst[1].y=0;
    coordinate_dst[2].x=0;
    coordinate_dst[2].y=height-1;
    coordinate_dst[3].x=width-1;
    coordinate_dst[3].y=height-1;
    cv::Mat M=cv::getPerspectiveTransform(points,coordinate_dst);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    if(use_bilinear){
        cv::cuda::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::cuda::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cv_cuda::cvtcolor(bool to_rgb){
    if(to_rgb==this->is_rgb){
        return copy();
    }
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::cvtColor(this->data.image,result->data.image,cv::COLOR_RGB2BGR);
    result->is_rgb=!this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

std::shared_ptr<QyImage> QyImage_cv_cuda::convertTo(QyImage::Data_type t){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> result;
    switch(t){
        case QyImage::Data_type::Float64:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_64FC(0));

        case QyImage::Data_type::Float32:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_32FC(0));
            break;

        case QyImage::Data_type::Float16:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_16FC(0));
            break;
        
        case QyImage::Data_type::Int32:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_32SC(0));
            break;
        case QyImage::Data_type::UInt16:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_16UC(0));
            break;
        case QyImage::Data_type::Int16:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_16SC(0));
            break;
        case QyImage::Data_type::UInt8:
            result=std::make_shared<QyImage_cv_cuda>(this->get_handle() );
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_8UC(0));
            break;
        default:

            break;
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

cv::Mat QyImage_cv_cuda::get_image(){
    cv::cuda::setDevice(this->data.device_idx);
    cv::Mat result;
    this->data.image.download(result);
    return result;
};

void QyImage_cv_cuda::set_image(cv::Mat input,bool is_rgb){
    cv::cuda::setDevice(this->data.device_idx);
    this->data.image.upload(input);
    this->is_rgb=is_rgb;
};

std::shared_ptr<QyImage> QyImage_cv_cuda::operator*(cv::Scalar value){
    cv::cuda::setDevice(this->data.device_idx);

    std::shared_ptr<QyImage_cv_cuda> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv_cuda* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv_cuda>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::multiply(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    result->data.device_idx=input->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_cv_cuda::operator/(cv::Scalar value){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv_cuda* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv_cuda>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::divide(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    result->data.device_idx=input->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv_cuda::operator+(cv::Scalar value){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv_cuda* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv_cuda>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::add(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    result->data.device_idx=input->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv_cuda::operator-(cv::Scalar value){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv_cuda* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv_cuda>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::subtract(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    result->data.device_idx=input->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv_cuda::scale_add(cv::Scalar factor,cv::Scalar value){
    cv::cuda::setDevice(this->data.device_idx);
    std::shared_ptr<QyImage_cv_cuda> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv_cuda* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv_cuda>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(this->get_handle());
    cv::cuda::GpuMat temp;
    cv::cuda::multiply(input->data.image,factor,temp);
    cv::cuda::add(temp,value,result->data.image);
    result->is_rgb=input->is_rgb;
    result->data.device_idx=input->data.device_idx;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};


std::shared_ptr<QyImage> from_data(uint8_t* img_data,int element_size,int src_stride,int width,int height,std::shared_ptr<Device_Handle> handle,bool is_rgb,bool from_host){
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(handle);
    result->data.image=cv::cuda::GpuMat(height,width,CV_8UC3);
    result->data.device_idx=handle->get_device_id();
    if(result->data.image.step!=src_stride){
        if(from_host){
            cudaMemcpy2D(result->data.image.data,result->data.image.step,img_data,src_stride,width*element_size,height,cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy2D(result->data.image.data,result->data.image.step,img_data,src_stride,width*element_size,height,cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        }
    }   
    else{
        if(from_host){
            cudaMemcpy(result->data.image.data,img_data,src_stride*height,cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(result->data.image.data,img_data,src_stride*height,cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }
    } 
    result->set_is_rgb(is_rgb);
    result->set_owned(true);
    return result;    
};

std::shared_ptr<QyImage> from_mat(cv::Mat img,std::shared_ptr<Device_Handle> handle,bool is_rgb){
    std::shared_ptr<QyImage_cv_cuda> result=std::make_shared<QyImage_cv_cuda>(handle);
    result->data.device_idx=handle->get_device_id();
    result->set_image(img,is_rgb);
    return result;
};

std::shared_ptr<QyImage> from_InputOutput(std::shared_ptr<InputOutput> src,bool copy){
    if(src->data_type!=InputOutput::Type::Image_t)
        return std::shared_ptr<QyImage>();
    if(copy){
        return src->data.image->copy();
    }
    else{
        return src->data.image;
    }

};
