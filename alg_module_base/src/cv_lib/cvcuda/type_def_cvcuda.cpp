#include "cv_lib/cvcuda/type_def_cvcuda.h"
#include <nvcv/TensorDataAccess.hpp>
#include "inout_type.h"

#ifdef CVCUDA_OLD            
    #define Size2D(x,y) nvcv::Size2D({x,y}) 
#else
    #define Size2D(x,y) nvcv::Size2D(x,y) 

#endif
std::vector<float> inverse(std::vector<float>& m){
    std::vector<float> result(6,0);
    double q=1.0/(m[0]*m[4]-m[1]*m[3]);
    result[0]=m[4]*q;
    result[1]=-m[1]*q;
    result[3]=-m[3]*q;
    result[4]=m[0]*q;
    result[2]=(m[1]*m[5]-m[4]*m[2])*q;
    result[5]=-(m[0]*m[5]-m[3]*m[2])*q;
    return result;
}

int QyImage_cvcuda::get_width(){
    return this->data.image->shape()[2];
};

int QyImage_cvcuda::get_height(){
    return this->data.image->shape()[1];    
};
bool QyImage_cvcuda::is_empty(){
    if(this->data.image->shape().empty())
        return true;
    for(int i=0;i<this->data.image->shape().size();i++){
        if(this->data.image->shape()[i]<=0)
            return true;
    }
    return false;
};


std::shared_ptr<QyImage> QyImage_cvcuda::copy(){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::ConvertTo op;
    switch(out){
        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
            break;
    }
    op(NULL,*(this->data.image),*(result->data.image),1,0);
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_cvcuda::crop(cv::Rect box){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::CustomCrop op;
    NVCVRectI rect;
    rect.x=box.x;
    rect.y=box.y;
    rect.width=box.width;
    rect.height=box.height;
    switch(out){

        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(box.width,box.height),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(box.width,box.height),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(box.width,box.height),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(box.width,box.height),nvcv::FMT_BGRf32);
            break;
    }
    op(NULL,*(this->data.image),*(result->data.image),rect);
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_cvcuda::resize(int width,int height,bool use_bilinear){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::Resize op;
    switch(out){
        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
    }
    if(use_bilinear)
        op(NULL,*(this->data.image),*(result->data.image),NVCVInterpolationType::NVCV_INTERP_LINEAR);
    else{
        op(NULL,*(this->data.image),*(result->data.image),NVCVInterpolationType::NVCV_INTERP_NEAREST);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;    
};
std::shared_ptr<QyImage> QyImage_cvcuda::crop_resize(cv::Rect box,int width,int height,bool use_bilinear){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage> temp=this->crop(box);
    std::shared_ptr<QyImage> result=temp->resize(width,height,use_bilinear);
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

std::shared_ptr<QyImage> QyImage_cvcuda::padding(int left,int right,int up,int down,int value){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::CopyMakeBorder op;
    int width=this->get_width()+left+right;
    int height=this->get_height()+up+down;
    switch(out){
        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
    }
    float4 v;
    v.x=value;
    v.y=value;
    v.z=value;
    v.w=value;
    op(NULL,*(this->data.image),*(result->data.image),up,left,NVCVBorderType::NVCV_BORDER_CONSTANT,v);
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;    

};

std::shared_ptr<QyImage> QyImage_cvcuda::warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::WarpAffine op(1);
    switch(out){
        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
    }
#ifdef CVCUDA_OLD            
    matrix=inverse(matrix);
#endif

    NVCVAffineTransform M;
    for(int i=0;i<6;i++){
        M[i]=matrix[i];
    }
    float4 v;
    v.x=0;
    v.y=0;
    v.z=0;
    v.w=0;


    try{

        throw 1;
    }
    catch(...){}
    if(use_bilinear){
        op(NULL,*(this->data.image),*(result->data.image),M,NVCVInterpolationType::NVCV_INTERP_LINEAR,NVCVBorderType::NVCV_BORDER_REPLICATE,v);
    }
    else{
        op(NULL,*(this->data.image),*(result->data.image),M,NVCVInterpolationType::NVCV_INTERP_NEAREST,NVCVBorderType::NVCV_BORDER_REPLICATE,v);
    }
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;    
};

std::shared_ptr<QyImage> QyImage_cvcuda::warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::WarpPerspective op(1);
    switch(out){

        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(width,height),nvcv::FMT_BGRf32);
            break;

    }
    NVCVPerspectiveTransform M;
    for(int i=0;i<9;i++){
        M[i]=matrix[i];
    }
    float4 v;
    v.x=0;
    v.y=0;
    v.z=0;
    v.w=0;

#ifndef CVCUDA_OLD            
    if(use_bilinear){
        op(NULL,*(this->data.image),*(result->data.image),M,NVCVInterpolationType::NVCV_INTERP_LINEAR,NVCVBorderType::NVCV_BORDER_REPLICATE,v);
    }
    else{
        op(NULL,*(this->data.image),*(result->data.image),M,NVCVInterpolationType::NVCV_INTERP_NEAREST,NVCVBorderType::NVCV_BORDER_REPLICATE,v);
    }
#else
    if(use_bilinear){
        op(NULL,*(this->data.image),*(result->data.image),M,NVCVInterpolationType::NVCV_INTERP_LINEAR|NVCVInterpolationType::NVCV_WARP_INVERSE_MAP,NVCVBorderType::NVCV_BORDER_REPLICATE,v);
    }
    else{
        op(NULL,*(this->data.image),*(result->data.image),M,NVCVInterpolationType::NVCV_INTERP_NEAREST|NVCVInterpolationType::NVCV_WARP_INVERSE_MAP,NVCVBorderType::NVCV_BORDER_REPLICATE,v);
    }
#endif

    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;    

};
std::shared_ptr<QyImage> QyImage_cvcuda::warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear){

    cudaSetDevice(handle->get_device_id());
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
    M.convertTo(M,CV_32FC1);
    std::vector<float> param;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            param.push_back(M.at<float>(i,j));
        }
    }

    return warp_perspective(param,width,height,use_bilinear);
};

std::shared_ptr<QyImage> QyImage_cvcuda::cvtcolor(bool to_rgb){
    if(to_rgb==is_rgb){
        return copy();
    }
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::CvtColor op;
    switch(out){
        case NVCV_DATA_TYPE_U8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf16);
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
            break;
    }
    op(NULL,*(this->data.image),*(result->data.image),NVCVColorConversionCode::NVCV_COLOR_BGR2RGB);
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;    

};

std::shared_ptr<QyImage> QyImage_cvcuda::convertTo(QyImage::Data_type t){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::ConvertTo op;
    switch(t){
        case QyImage::Data_type::UInt8: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGR8);
            break;
#ifndef CVCUDA_OLD            
        case QyImage::Data_type::Float16: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf16);
            break;
#endif            
        case QyImage::Data_type::Float32: 
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
            break;
        default:
            result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
            break;
    }
    op(NULL,*(this->data.image),*(result->data.image),1,0);
    result->is_rgb=this->is_rgb;
    result->data.device_idx=this->handle->get_device_id();
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

cv::Mat QyImage_cvcuda::get_image(){
    cudaSetDevice(handle->get_device_id());
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);
    cvcuda::CvtColor op;
    cv::Mat output;
    nvcv::ImageFormat fmt;
    QyImage_cvcuda* input=this;
    std::shared_ptr<QyImage_cvcuda> cache;
    int pixel_byte=3;
    switch(out){
        case NVCV_DATA_TYPE_U8: 
            output=cv::Mat(get_height(),get_width(),CV_8UC(3),cv::Scalar(0,0,0));
            fmt=nvcv::FMT_BGR8;
            pixel_byte=3;
            break;
#ifndef CVCUDA_OLD            
        case NVCV_DATA_TYPE_F16: 
            output=cv::Mat(get_height(),get_width(),CV_16FC(3),cv::Scalar(0,0,0));
            fmt=nvcv::FMT_BGRAf16;
            pixel_byte=6;
            break;
#endif            
        case NVCV_DATA_TYPE_F32: 
            output=cv::Mat(get_height(),get_width(),CV_32FC(3),cv::Scalar(0,0,0));
            fmt=nvcv::FMT_BGRAf32;
            pixel_byte=12;
            break;
        default:
            cache=std::dynamic_pointer_cast<QyImage_cvcuda>(input->convertTo(QyImage::Data_type::Float32));
            input=cache.get();
            output=cv::Mat(get_height(),get_width(),CV_32FC(3),cv::Scalar(0,0,0));
            fmt=nvcv::FMT_BGRAf32;
            pixel_byte=12;
            break;
    }
    nvcv::Tensor::Requirements inReqs=nvcv::Tensor::CalcRequirements(input->data.image->shape(),input->data.image->dtype());
#ifdef CVCUDA_OLD
    auto temp_base=input->data.image->exportData();
    auto temp_out=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto temp_out=input->data.image->exportData<nvcv::TensorDataStridedCuda>();
#endif

//    cudaMemcpy(output.data,(uint8_t*)temp_out->basePtr(),output.rows*output.cols*output.elemSize(),cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaMemcpy2D(output.data,get_width()*pixel_byte,(uint8_t*)temp_out->basePtr(),inReqs.strides[1],get_width()*pixel_byte,get_height(),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    return output;
};
void QyImage_cvcuda::set_image(cv::Mat input,bool is_rgb){
    cudaSetDevice(handle->get_device_id());
    nvcv::ImageFormat dst_type;
    int type=input.type()&CV_MAT_DEPTH_MASK;
    int bit=1;
    switch(type){
        case CV_8U:
            dst_type=nvcv::FMT_BGR8;
            bit=1;
            break;
#ifndef CVCUDA_OLD            
        case CV_16F:
            dst_type=nvcv::FMT_BGRf16;
            bit=2;
            break;
#endif
        case CV_32F:
            dst_type=nvcv::FMT_BGRf32;
            bit=4;
            break;
        default:
            input.convertTo(input,CV_32FC(3));
            dst_type=nvcv::FMT_BGRf32;
            bit=4;
            break;
    }
    int width=input.cols;
    int height=input.rows;
    nvcv::Tensor::Requirements inReqs=nvcv::Tensor::CalcRequirements(1,{width,height},dst_type);

    this->data.image=std::make_shared<nvcv::Tensor>(inReqs);
#ifdef CVCUDA_OLD
    auto temp_base=this->data.image->exportData();
    auto temp=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto temp=this->data.image->exportData<nvcv::TensorDataStridedCuda>();
#endif
    cudaMemcpy2D((uint8_t*)temp->basePtr(),inReqs.strides[1],input.data,width*input.elemSize(),width*input.elemSize(),input.rows,cudaMemcpyKind::cudaMemcpyHostToDevice);
    this->is_rgb=is_rgb;

};


std::shared_ptr<QyImage> QyImage_cvcuda::operator*(cv::Scalar value){
    cudaSetDevice(handle->get_device_id());

    std::shared_ptr<QyImage_cvcuda> cache;
    QyImage_cvcuda* input=this;
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);

    if(out!=NVCV_DATA_TYPE_F32){
        cache=std::dynamic_pointer_cast<QyImage_cvcuda>(this->convertTo(QyImage::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }

    nvcvTensorGetDataType(input->data.image->handle(), &out);
    
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    cvcuda::Normalize op;
    nvcv::ImageFormat select_format;
    result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
    select_format=nvcv::FMT_BGRf32;
    nvcv::Tensor value_tensor=nvcv::Tensor({{1,1,1,select_format.numChannels()},nvcv::TENSOR_NHWC},select_format.planeDataType(0));
    nvcv::Tensor factor_tensor=nvcv::Tensor({{1,1,1,select_format.numChannels()},nvcv::TENSOR_NHWC},select_format.planeDataType(0));

    std::vector<float> factor_vec(3,1);
    std::vector<float> value_vec(3,0);
    for(int i=0;i<3;i++){
        factor_vec[i]=float(value.val[i]);   
    }

    {
#ifdef CVCUDA_OLD
    auto temp_base=value_tensor.exportData();
    auto baseData=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto baseData=value_tensor.exportData<nvcv::TensorDataStridedCuda>();
#endif
        auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
        cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), value_vec.data(),
                                            value_vec.size() * sizeof(float),
                                            value_vec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice);
    }

    {
#ifdef CVCUDA_OLD
    auto temp_base=factor_tensor.exportData();
    auto baseData=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto baseData=factor_tensor.exportData<nvcv::TensorDataStridedCuda>();
#endif
        auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
        cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), factor_vec.data(),
                                            factor_vec.size() * sizeof(float),
                                            factor_vec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice);
    }

    op(NULL,*(input->data.image),value_tensor,factor_tensor,*(result->data.image),1,0,0);
    result->data.device_idx=input->handle->get_device_id();
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_cvcuda::operator/(cv::Scalar value){
    value.val[0]=1.0/value.val[0];
    value.val[1]=1.0/value.val[1];
    value.val[2]=1.0/value.val[2];
    return this->operator*(value);

};
std::shared_ptr<QyImage> QyImage_cvcuda::operator+(cv::Scalar value){
    value=cv::Scalar(0,0,0)-value;
    return this->operator-(value);

};
std::shared_ptr<QyImage> QyImage_cvcuda::operator-(cv::Scalar value){
    std::shared_ptr<QyImage_cvcuda> cache;
    QyImage_cvcuda* input=this;
    NVCVDataType out;
    nvcvTensorGetDataType(this->data.image->handle(), &out);

    if(out!=NVCV_DATA_TYPE_F32){
        cache=std::dynamic_pointer_cast<QyImage_cvcuda>(this->convertTo(QyImage::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }

    nvcvTensorGetDataType(input->data.image->handle(), &out);
    
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(this->get_handle());
    cvcuda::Normalize op;
    nvcv::ImageFormat select_format;
    result->data.image=std::make_shared<nvcv::Tensor>(1,Size2D(get_width(),get_height()),nvcv::FMT_BGRf32);
    select_format=nvcv::FMT_BGRf32;
    nvcv::Tensor value_tensor=nvcv::Tensor({{1,1,1,select_format.numChannels()},nvcv::TENSOR_NHWC},select_format.planeDataType(0));
    nvcv::Tensor factor_tensor=nvcv::Tensor({{1,1,1,select_format.numChannels()},nvcv::TENSOR_NHWC},select_format.planeDataType(0));

    std::vector<float> factor_vec(3,1);
    std::vector<float> value_vec(3,0);
    for(int i=0;i<3;i++){
        value_vec[i]=float(value.val[i]);   
    }

    {
#ifdef CVCUDA_OLD
    auto temp_base=value_tensor.exportData();
    auto baseData=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto baseData=value_tensor.exportData<nvcv::TensorDataStridedCuda>();
#endif
        auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
        cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), value_vec.data(),
                                            value_vec.size() * sizeof(float),
                                            value_vec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice);
    }

    {
#ifdef CVCUDA_OLD
    auto temp_base=factor_tensor.exportData();
    auto baseData=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto baseData=factor_tensor.exportData<nvcv::TensorDataStridedCuda>();
#endif
        auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
        cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), factor_vec.data(),
                                            factor_vec.size() * sizeof(float),
                                            factor_vec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice);
    }

    op(NULL,*(input->data.image),value_tensor,factor_tensor,*(result->data.image),1,0,0);
    result->data.device_idx=input->handle->get_device_id();
    result->is_rgb=input->is_rgb;
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_cvcuda::scale_add(cv::Scalar factor,cv::Scalar value){
    std::shared_ptr<QyImage> temp=this->operator*(factor);
    if(temp==nullptr)
        return temp;
    std::shared_ptr<QyImage_cvcuda> temp1=std::dynamic_pointer_cast<QyImage_cvcuda>(temp);
    return temp1->operator+(value);

};

std::shared_ptr<QyImage> from_data(uint8_t* img_data,int element_size,int src_stride,int width,int height,std::shared_ptr<Device_Handle> handle,bool is_rgb,bool from_host){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(handle);

    nvcv::Tensor::Requirements inReqs=nvcv::Tensor::CalcRequirements(1,{width,height},nvcv::FMT_BGR8);
    result->data.image=std::make_shared<nvcv::Tensor>(inReqs);
    result->data.device_idx=handle->get_device_id();
    result->set_is_rgb(is_rgb);
    result->set_owned(true);
#ifdef CVCUDA_OLD
    auto temp_base=result->data.image->exportData();
    auto temp=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto temp=result->data.image->exportData<nvcv::TensorDataStridedCuda>();
#endif
    if(inReqs.strides[1]!=src_stride){
        if(from_host){
            cudaMemcpy2D((uint8_t*)temp->basePtr(),inReqs.strides[1],img_data,src_stride,width*element_size,height,cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy2D((uint8_t*)temp->basePtr(),inReqs.strides[1],img_data,src_stride,width*element_size,height,cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        }
    }   
    else{
        if(from_host){
            cudaMemcpy((uint8_t*)temp->basePtr(),img_data,src_stride*height,cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy((uint8_t*)temp->basePtr(),img_data,src_stride*height,cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }
    } 
    return result;    
    
};

std::shared_ptr<QyImage> from_mat(cv::Mat img,std::shared_ptr<Device_Handle> handle,bool is_rgb){
    cudaSetDevice(handle->get_device_id());
    std::shared_ptr<QyImage_cvcuda> result=std::make_shared<QyImage_cvcuda>(handle);
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
