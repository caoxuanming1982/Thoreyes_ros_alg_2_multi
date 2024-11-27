#include "cv_lib/cv/type_def_cv.h"
#include "inout_type.h"

int QyImage_cv::get_width(){
    return this->data.image.cols;
};
int QyImage_cv::get_height(){
    return this->data.image.rows;
};
bool QyImage_cv::is_empty(){
    if(data.image.cols<=0||data.image.rows<=0)
        return true;
    return false;
};


std::shared_ptr<QyImage> QyImage_cv::copy(){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    result->data.image=this->data.image.clone();
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_cv::crop(cv::Rect box){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    result->data.image=this->data.image(box).clone();
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv::resize(int width,int height,bool use_bilinear){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    if(use_bilinear){
        cv::resize(this->data.image,result->data.image,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::resize(this->data.image,result->data.image,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv::crop_resize(cv::Rect box,int width,int height,bool use_bilinear){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::Mat temp=this->data.image(box);
    if(use_bilinear){
        cv::resize(temp,result->data.image,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::resize(temp,result->data.image,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cv::padding(int left,int right,int up,int down,int value){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::copyMakeBorder(this->data.image,result->data.image,up,down,left,right,cv::BORDER_CONSTANT,cv::Scalar(value,value,value));
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cv::warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::Mat M(2,3,CV_32FC1);
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            M.at<float>(i,j)=matrix[i*3+j];            
        }
    }
    if(use_bilinear){
        cv::warpAffine(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_LINEAR,cv::BORDER_REPLICATE);

    }
    else{
        cv::warpAffine(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_NEAREST,cv::BORDER_REPLICATE);
    }
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

std::shared_ptr<QyImage> QyImage_cv::warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::Mat M(3,3,CV_32FC1);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            M.at<float>(i,j)=matrix[i*3+j];            
        }
    }
    if(use_bilinear){
        cv::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv::warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear){
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
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    if(use_bilinear){
        cv::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::warpPerspective(this->data.image,result->data.image,M,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cv::cvtcolor(bool to_rgb){
    if(to_rgb==this->is_rgb){
        return copy();
    }
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::cvtColor(this->data.image,result->data.image,cv::COLOR_RGB2BGR);
    result->is_rgb=!this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

std::shared_ptr<QyImage> QyImage_cv::convertTo(QyImage::Data_type t){
    std::shared_ptr<QyImage_cv> result;
    switch(t){
        case QyImage::Data_type::Float64:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_64FC(0));

        case QyImage::Data_type::Float32:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_32FC(0));
            break;

#ifndef USE_IX            
        case QyImage::Data_type::Float16:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_16FC(0));
            break;
#endif        
        case QyImage::Data_type::Int32:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_32SC(0));
            break;
        case QyImage::Data_type::UInt16:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_16UC(0));
            break;
        case QyImage::Data_type::Int16:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_16SC(0));
            break;
        case QyImage::Data_type::UInt8:
            result=std::make_shared<QyImage_cv>(this->get_handle());
            this->data.image.convertTo(result->data.image,this->data.image.depth()&~CV_MAT_DEPTH_MASK+CV_8UC(0));
            break;
        default:

            break;
    }
    result->is_rgb=this->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

cv::Mat QyImage_cv::get_image(){
    return this->data.image.clone();
};

void QyImage_cv::set_image(cv::Mat input,bool is_rgb){
    this->data.image=input;
    this->is_rgb=is_rgb;
};

std::shared_ptr<QyImage> QyImage_cv::operator*(cv::Scalar value){
    std::shared_ptr<QyImage_cv> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::multiply(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_cv::operator/(cv::Scalar value){
    std::shared_ptr<QyImage_cv> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::divide(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv::operator+(cv::Scalar value){
    std::shared_ptr<QyImage_cv> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::add(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv::operator-(cv::Scalar value){
    std::shared_ptr<QyImage_cv> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::subtract(input->data.image,value,result->data.image);
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cv::scale_add(cv::Scalar factor,cv::Scalar value){
    std::shared_ptr<QyImage_cv> cache;
    int type = this->data.image.type() & CV_MAT_DEPTH_MASK;
    QyImage_cv* input=this;
    if (type != CV_32F)
    {
        cache=std::dynamic_pointer_cast<QyImage_cv>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(this->get_handle());
    cv::Mat temp;
    cv::multiply(input->data.image,factor,temp);
    cv::add(temp,value,result->data.image);
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

std::shared_ptr<QyImage> from_data(uint8_t* img_data,int element_size,int src_stride,int width,int height,std::shared_ptr<Device_Handle> handle,bool is_rgb,bool from_host){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(handle);
    if(element_size*width<src_stride){
        result->data.image=cv::Mat(height,width,CV_8UC3);
        for(int i=0;i<height;i++){
            memcpy(result->data.image.data+ i*width*element_size,img_data+i*src_stride,i*width*element_size);
        }
    }   
    else{
        memcpy(result->data.image.data,img_data,height*width*element_size);
    } 
    result->set_is_rgb(is_rgb);
    result->set_owned(true);
    return result;
};


std::shared_ptr<QyImage> from_mat(cv::Mat img,std::shared_ptr<Device_Handle> handle,bool is_rgb){
    std::shared_ptr<QyImage_cv> result=std::make_shared<QyImage_cv>(handle);
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
