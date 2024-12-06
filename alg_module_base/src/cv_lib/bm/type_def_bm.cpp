#include "cv_lib/bm/type_def_bm.h"
#include "inout_type.h"
#include "network_engine/bm/device_handle_bm.h"

int QyImage_bm::get_width(){
    return this->data.image.width;
};
int QyImage_bm::get_height(){
    return this->data.image.height;
};

bool QyImage_bm::is_empty(){
    if(bm_image_is_attached(this->data.image)&&get_width()>0&&get_height()>0){
        return false;
    }
    return true;
};


std::shared_ptr<QyImage> QyImage_bm::copy(){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,get_height(),get_width(),this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    result->owned=true;
    result->is_rgb=this->is_rgb;
//    bmcv_rect_t box_bm;
//    box_bm.start_x=0;
 //   box_bm.start_y=0;
  //  box_bm.crop_w=get_width();
   // box_bm.crop_h=get_height();
    bm_image_alloc_dev_mem(result->data.image);

    bmcv_copy_to_atrr_t box;
    box.start_x=0;
    box.start_y=0;

    bmcv_image_copy_to(result->data.handle,box,this->data.image, (result->data.image));
//    bmcv_image_vpp_convert(result->data.handle, 1, this->data.image, &(result->data.image), &box_bm);
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};


std::shared_ptr<QyImage> QyImage_bm::crop(cv::Rect box){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,box.height,box.width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_rect_t box_bm;
    box_bm.start_x=box.x;
    box_bm.start_y=box.y;
    box_bm.crop_w=box.width;
    box_bm.crop_h=box.height;
#if 0
    if(box_bm.crop_w <= 0 ){
        box_bm.crop_w = this->data.image.width;
    }
    if(box_bm.crop_h <= 0 ){
        box_bm.crop_h = this->data.image.height;
    }
#endif
    bm_image_alloc_dev_mem(result->data.image);
    bmcv_image_vpp_convert(result->data.handle, 1, this->data.image, &(result->data.image), &box_bm);
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_bm::resize(int width,int height,bool use_bilinear){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,height,width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_rect_t box_bm;
    box_bm.start_x=0;
    box_bm.start_y=0;
    box_bm.crop_w=this->get_width();
    box_bm.crop_h=this->get_height();
#if 0
    if(box_bm.crop_w <= 0 ){
        box_bm.crop_w = this->data.image.width;
    }
    if(box_bm.crop_h <= 0 ){
        box_bm.crop_h = this->data.image.height;
    }	
#endif
    if(use_bilinear){
        bmcv_image_vpp_convert(result->data.handle, 1, this->data.image, &(result->data.image), &box_bm);
    }
    else{
        bmcv_image_vpp_convert(result->data.handle, 1, this->data.image, &(result->data.image), &box_bm,BMCV_INTER_NEAREST);

    }
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_bm::crop_resize(cv::Rect box,int width,int height,bool use_bilinear){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,height,width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_rect_t box_bm;
    box_bm.start_x=box.x;
    box_bm.start_y=box.y;
    box_bm.crop_w=box.width;
    box_bm.crop_h=box.height;
#if 0
    if(box_bm.crop_w <= 0 ){
        box_bm.crop_w = this->data.image.width;
    }
    if(box_bm.crop_h <= 0 ){
        box_bm.crop_h = this->data.image.height;
    }
#endif
    if(use_bilinear){
        bmcv_image_vpp_convert(result->data.handle, 1, this->data.image, &(result->data.image), &box_bm);
    }
    else{
        bmcv_image_vpp_convert(result->data.handle, 1, this->data.image, &(result->data.image), &box_bm,BMCV_INTER_NEAREST);
    }
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};


std::shared_ptr<QyImage> QyImage_bm::padding(int left,int right,int up,int down,int value){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    int width=this->get_width()+left+right;
    int height=this->get_height()+up+down;
    bm_image_create(result->data.handle,height,width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_copy_to_atrr_t offset;
    offset.start_x=left;
    offset.start_y=up;
    offset.if_padding=true;
    offset.padding_b=value;
    offset.padding_g=value;
    offset.padding_r=value;
    bmcv_image_copy_to(result->data.handle,offset,this->data.image,result->data.image);
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_bm::warp_affine(std::vector<float>& matrix_in,int width,int height,bool use_bilinear){
    bmcv_affine_image_matrix matrix;
    matrix.matrix_num=1;
    matrix.matrix=new bmcv_warp_matrix[1];
    for(int i=0;i<6;i++){
        matrix.matrix->m[i]=matrix_in[i];
    }

    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,height,width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_image_warp_affine_similar_to_opencv(result->data.handle,1,&matrix,&(this->data.image),&(result->data.image),use_bilinear);
    std::shared_ptr<QyImage> result_out=result;
    delete[] matrix.matrix;
    matrix.matrix=nullptr;
    return result_out;

};

std::shared_ptr<QyImage> QyImage_bm::warp_perspective(std::vector<float>& matrix_in,int width,int height,bool use_bilinear){
    bmcv_perspective_image_matrix matrix;
    matrix.matrix_num=1;
    matrix.matrix=new bmcv_perspective_matrix[1];
    for(int i=0;i<9;i++){
        matrix.matrix->m[i]=matrix_in[i];
    }

    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,height,width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_image_warp_perspective_similar_to_opencv(result->data.handle,1,&matrix,&(this->data.image),&(result->data.image),use_bilinear);
    std::shared_ptr<QyImage> result_out=result;
    delete[] matrix.matrix;
    matrix.matrix=nullptr;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_bm::warp_perspective(std::vector<cv::Point2f>& points_in,int width,int height,bool use_bilinear){
    std::vector<cv::Point2f> coordinate_dst(4);
    coordinate_dst[0].x=0;
    coordinate_dst[0].y=0;
    coordinate_dst[1].x=width-1;
    coordinate_dst[1].y=0;
    coordinate_dst[2].x=0;
    coordinate_dst[2].y=height-1;
    coordinate_dst[3].x=width-1;
    coordinate_dst[3].y=height-1;
    std::vector<cv::Point2f> points_in_t=points_in;
    for(int i=0;i<points_in.size();i++){
        points_in_t[i].x+=0.5;
        points_in_t[i].y+=0.5;
    }
    cv::Mat M=cv::getPerspectiveTransform(points_in_t,coordinate_dst);
    M.convertTo(M,CV_32FC1);
    std::vector<float> param;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            param.push_back(M.at<float>(i,j));
        }
    }
    return warp_perspective(param,width,height,use_bilinear);


/*
    bmcv_perspective_image_coordinate coord;
    coord.coordinate_num=1;
    coord.coordinate=new bmcv_perspective_coordinate[1];
    for(int i=0;i<4;i++){
        coord.coordinate[0].x[i]=points_in[i].x;
        coord.coordinate[0].y[i]=points_in[i].y;
    }
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
    result->data.handle=this->data.handle;
    bm_image_create(result->data.handle,height,width,this->data.image.image_format,this->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=this->is_rgb;
    bmcv_image_warp_perspective_with_coordinate(result->data.handle,1,&coord,&(this->data.image),&(result->data.image),use_bilinear);
    std::shared_ptr<QyImage> result_out=result;
    delete[] coord.coordinate;
    coord.coordinate=nullptr;
    return result_out;
*/    
};

std::shared_ptr<QyImage> QyImage_bm::cvtcolor(bool to_rgb){
    if(RGB==to_rgb){
        return this->copy();
    }
    else{

        bm_image_format_ext dst_type=bm_image_format_ext::FORMAT_BGR_PLANAR;
        if(src_channel_exchange_dst_channel.find(this->data.image.image_format)!=src_channel_exchange_dst_channel.end()){
            dst_type=src_channel_exchange_dst_channel[this->data.image.image_format];
        }
        else{
            return std::shared_ptr<QyImage>();
        }

        std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
        result->data.handle=this->data.handle;
        result->owned=true;
        result->is_rgb=!this->is_rgb;
        
        bm_image_create(result->data.handle,get_height(),get_width(),dst_type,this->data.image.data_type,&(result->data.image));
        bm_image_alloc_dev_mem(result->data.image);
        bmcv_image_storage_convert(result->data.handle,1,&(this->data.image),&(result->data.image));
        std::shared_ptr<QyImage> result_out=result;
        return result_out;
    }

};

std::shared_ptr<QyImage> QyImage_bm::convertTo(QyImage::Data_type t){
    std::shared_ptr<QyImage_bm> result;
    switch(t){
        case QyImage::Data_type::Float32:
            result=std::make_shared<QyImage_bm>(this->get_handle());
            result->data.handle=this->data.handle;
            bm_image_create(result->data.handle,get_height(),get_width(),this->data.image.image_format,bm_image_data_format_ext::DATA_TYPE_EXT_FLOAT32,&(result->data.image));
            bm_image_alloc_dev_mem(result->data.image);
            result->owned=true;
            result->is_rgb=this->is_rgb;
            bmcv_image_storage_convert(result->data.handle,1,&(this->data.image),&(result->data.image));
            break;
        
        case QyImage::Data_type::UInt8:
            result=std::make_shared<QyImage_bm>(this->get_handle());
            result->data.handle=this->data.handle;
            bm_image_create(result->data.handle,get_height(),get_width(),this->data.image.image_format,bm_image_data_format_ext::DATA_TYPE_EXT_1N_BYTE,&(result->data.image));
            bm_image_alloc_dev_mem(result->data.image);
            result->owned=true;
            result->is_rgb=this->is_rgb;
            bmcv_image_storage_convert(result->data.handle,1,&(this->data.image),&(result->data.image));
            break;

        case QyImage::Data_type::Float16:
            result=std::make_shared<QyImage_bm>(this->get_handle());
            result->data.handle=this->data.handle;
            bm_image_create(result->data.handle,get_height(),get_width(),this->data.image.image_format,bm_image_data_format_ext::DATA_TYPE_EXT_FP16,&(result->data.image));
            bm_image_alloc_dev_mem(result->data.image);
            result->owned=true;
            result->is_rgb=this->is_rgb;
            bmcv_image_storage_convert(result->data.handle,1,&(this->data.image),&(result->data.image));
            break;

        default:

            break;
    }
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};

cv::Mat QyImage_bm::get_image(){
    bm_image input_cache;
    cv::Mat output;
    switch (this->data.image.data_type)
    {
        case DATA_TYPE_EXT_1N_BYTE:
            bm_image_create(this->data.handle, get_height(),get_width(),FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, &input_cache);	
            output=cv::Mat(get_height(),get_width(),CV_8UC(3),cv::Scalar(0,0,0));
            break;
        case DATA_TYPE_EXT_FP16:
            bm_image_create(this->data.handle, get_height(),get_width(),FORMAT_BGR_PACKED, DATA_TYPE_EXT_FP16, &input_cache);	
            output=cv::Mat(get_height(),get_width(),CV_16FC(3),cv::Scalar(0,0,0));
            break;

        case DATA_TYPE_EXT_FLOAT32:
            bm_image_create(this->data.handle, get_height(),get_width(),FORMAT_BGR_PACKED, DATA_TYPE_EXT_FLOAT32, &input_cache);	
            output=cv::Mat(get_height(),get_width(),CV_32FC(3),cv::Scalar(0,0,0));
            break;

    
        default:
            bm_image_create(this->data.handle, get_height(),get_width(),FORMAT_BGR_PACKED, DATA_TYPE_EXT_FLOAT32, &input_cache);	
            output=cv::Mat(get_height(),get_width(),CV_32FC(3),cv::Scalar(0,0,0));
            break;
    }
    bm_image_alloc_dev_mem(input_cache);
    bmcv_image_storage_convert(this->data.handle, 1, &(this->data.image), &input_cache);
    void* ptr1=(void*)output.ptr<int8_t>();
    bm_image_copy_device_to_host(input_cache,&ptr1);
    bm_image_destroy(input_cache);
    return output;
};

void QyImage_bm::set_image(cv::Mat input,bool is_rgb){
    bool need_alloc=false;
    this->is_rgb=is_rgb;
    if(bm_image_is_attached(this->data.image)){
        if(get_width()!=input.cols||get_height()!=input.rows){
            need_alloc=true;
            if(this->owned){
                bm_image_destroy(this->data.image);
            }
            else{
                bm_image_detach(this->data.image);
            }
        }
    }
    else{
        need_alloc=true;
    }
    bm_image_data_format_ext dst_type=DATA_TYPE_EXT_1N_BYTE;
    int type=input.type()&CV_MAT_DEPTH_MASK;
    switch(type){
        case CV_8U:
            dst_type=DATA_TYPE_EXT_1N_BYTE;
            break;
        case CV_16F:
            dst_type=DATA_TYPE_EXT_FP16;
            break;

        case CV_32F:
            dst_type=DATA_TYPE_EXT_FLOAT32;
            break;
        default:
            dst_type=DATA_TYPE_EXT_FLOAT32;
            input.convertTo(input,input.depth()&~CV_MAT_DEPTH_MASK+CV_32FC(0));

    }


    int data_length=input.cols*input.rows*input.elemSize();
    bm_image temp;
        if(is_rgb){
            bm_image_create(this->data.handle, input.rows,input.cols,FORMAT_RGB_PACKED, DATA_TYPE_EXT_1N_BYTE, &temp);	
        }
        else{
            bm_image_create(this->data.handle, input.rows,input.cols,FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, &temp);
        }
        bm_image_alloc_dev_mem(temp);

    if(need_alloc){
        if(is_rgb){
            bm_image_create(this->data.handle, input.rows,input.cols,src_type2dst_type[FORMAT_RGB_PACKED], DATA_TYPE_EXT_1N_BYTE, &this->data.image);	
        }
        else{
            bm_image_create(this->data.handle, input.rows,input.cols,src_type2dst_type[FORMAT_BGR_PACKED], DATA_TYPE_EXT_1N_BYTE, &this->data.image);
        }
        bm_image_alloc_dev_mem(this->data.image);

    }

    void* ptr1=(void*)input.ptr<int8_t>();
    this->owned=true;
    bm_image_copy_host_to_device(temp,&ptr1);
    bmcv_image_storage_convert(this->data.handle,1,&temp,&(this->data.image));
    bm_image_destroy(temp);
};

std::shared_ptr<QyImage> QyImage_bm::to_HWC(){
    std::shared_ptr<QyImage> result_output;
    if(this->chw2hwc_map.find(this->data.image.image_format)!=this->chw2hwc_map.end()){
        std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
        result->data.handle=this->data.handle;
        bm_image_create(result->data.handle,get_height(),get_width(),chw2hwc_map[this->data.image.image_format],this->data.image.data_type,&(result->data.image));
        bm_image_alloc_dev_mem(result->data.image);
        result->owned=true;
        result->is_rgb=this->is_rgb;
        bmcv_image_storage_convert(result->data.handle,1,&(this->data.image),&(result->data.image));
        result_output=result;
    }
    else if(this->hwc2chw_map.find(this->data.image.image_format)!=this->hwc2chw_map.end()) {
        result_output=this->copy();
    }
    return result_output;
};
std::shared_ptr<QyImage> QyImage_bm::to_CHW(){
    std::shared_ptr<QyImage> result_output;
    if(this->hwc2chw_map.find(this->data.image.image_format)!=this->hwc2chw_map.end()){
        std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(this->get_handle());
        result->data.handle=this->data.handle;
        bm_image_create(result->data.handle,get_height(),get_width(),hwc2chw_map[this->data.image.image_format],this->data.image.data_type,&(result->data.image));
        bm_image_alloc_dev_mem(result->data.image);
        result->owned=true;
        result->is_rgb=this->is_rgb;
        bmcv_image_storage_convert(result->data.handle,1,&(this->data.image),&(result->data.image));
        result_output=result;
    }
    else if(this->chw2hwc_map.find(this->data.image.image_format)!=this->chw2hwc_map.end()) {
        result_output=this->copy();
    }
    return result_output;

};
std::shared_ptr<QyImage> QyImage_bm::auto_swap_HWC(Shape_t shape){

    if(shape.dims[shape.num_dims-1]==3){

        return this->to_HWC();

    }
    else if (shape.dims[shape.num_dims-3]==3){

        return this->to_CHW();

    }
    else{
        return this->copy();
    }
};

std::shared_ptr<QyImage> QyImage_bm::operator*(cv::Scalar value){
    std::shared_ptr<QyImage_bm> cache;
    bm_image_data_format_ext type = this->data.image.data_type;
    QyImage_bm* input=this;
    if (type != DATA_TYPE_EXT_FLOAT32)
    {
        cache=std::dynamic_pointer_cast<QyImage_bm>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    


    bmcv_convert_to_attr param;
    param.alpha_0=(float)value.val[0];
    param.alpha_1=(float)value.val[1];
    param.alpha_2=(float)value.val[2];
    param.beta_0=0;
    param.beta_1=0;
    param.beta_2=0;
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(input->get_handle());
    result->data.handle=input->data.handle;
    bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=input->is_rgb;
    bmcv_image_convert_to(result->data.handle, 1,param, &(input->data.image), &(result->data.image));
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_bm::operator/(cv::Scalar value){
    std::shared_ptr<QyImage_bm> cache;
    bm_image_data_format_ext type = this->data.image.data_type;
    QyImage_bm* input=this;
    if (type != DATA_TYPE_EXT_FLOAT32)
    {
        cache=std::dynamic_pointer_cast<QyImage_bm>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    bmcv_convert_to_attr param;
    param.alpha_0=1.0/(float)value.val[0];
    param.alpha_1=1.0/(float)value.val[1];
    param.alpha_2=1.0/(float)value.val[2];
    param.beta_0=0;
    param.beta_1=0;
    param.beta_2=0;
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(input->get_handle());
    result->data.handle=input->data.handle;
    bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=input->is_rgb;
    bmcv_image_convert_to(result->data.handle, 1,param, &(input->data.image), &(result->data.image));
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_bm::operator+(cv::Scalar value){
    std::shared_ptr<QyImage_bm> cache;
    bm_image_data_format_ext type = this->data.image.data_type;
    QyImage_bm* input=this;
    if (type != DATA_TYPE_EXT_FLOAT32)
    {
        cache=std::dynamic_pointer_cast<QyImage_bm>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    bmcv_convert_to_attr param;
    param.alpha_0=1;
    param.alpha_1=1;
    param.alpha_2=1;
    param.beta_0=(float)value.val[0];
    param.beta_1=(float)value.val[1];
    param.beta_2=(float)value.val[2];
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(input->get_handle());
    result->data.handle=input->data.handle;
    bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=input->is_rgb;
    if(value.val[0]!=0 && value.val[1]!=0 && value.val[2]!=0 ){
        bm_image temp_image;
        bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&temp_image);
        bm_image_alloc_dev_mem(temp_image);
        bmcv_convert_to_attr param1=param;
        param1.alpha_2=1;
        param1.beta_2=0;
        bmcv_image_convert_to(result->data.handle, 1,param1, &(input->data.image), &temp_image);

        bmcv_convert_to_attr param2=param;
        param2.alpha_0=1;
        param2.alpha_1=1;
        param2.beta_0=0;
        param2.beta_1=0;
        bmcv_image_convert_to(result->data.handle, 1,param2, &temp_image, &(result->data.image));
        bm_image_destroy(temp_image);
    }
    else{
        bmcv_image_convert_to(result->data.handle, 1,param, &(input->data.image), &(result->data.image));
    }

    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_bm::operator-(cv::Scalar value){
    std::shared_ptr<QyImage_bm> cache;
    bm_image_data_format_ext type = this->data.image.data_type;
    QyImage_bm* input=this;
    if (type != DATA_TYPE_EXT_FLOAT32)
    {
        cache=std::dynamic_pointer_cast<QyImage_bm>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    bmcv_convert_to_attr param;
    param.alpha_0=1;
    param.alpha_1=1;
    param.alpha_2=1;
    param.beta_0=-(float)value.val[0];
    param.beta_1=-(float)value.val[1];
    param.beta_2=-(float)value.val[2];
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(input->get_handle());
    result->data.handle=input->data.handle;
    bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=input->is_rgb;
    if(value.val[0]!=0 && value.val[1]!=0 && value.val[2]!=0 ){
        bm_image temp_image;
        bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&temp_image);
        bm_image_alloc_dev_mem(temp_image);
        bmcv_convert_to_attr param1=param;
        param1.alpha_2=1;
        param1.beta_2=0;
        bmcv_image_convert_to(result->data.handle, 1,param1, &(input->data.image), &temp_image);

        bmcv_convert_to_attr param2=param;
        param2.alpha_0=1;
        param2.alpha_1=1;
        param2.beta_0=0;
        param2.beta_1=0;
        bmcv_image_convert_to(result->data.handle, 1,param2, &temp_image, &(result->data.image));
        bm_image_destroy(temp_image);
    }
    else{
        bmcv_image_convert_to(result->data.handle, 1,param, &(input->data.image), &(result->data.image));
    }

    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};
std::shared_ptr<QyImage> QyImage_bm::scale_add(cv::Scalar factor,cv::Scalar value){
    std::shared_ptr<QyImage_bm> cache;
    bm_image_data_format_ext type = this->data.image.data_type;
    QyImage_bm* input=this;
    if (type != DATA_TYPE_EXT_FLOAT32)
    {
        cache=std::dynamic_pointer_cast<QyImage_bm>(this->convertTo(QyImage::Data_type::Float32));
        input=cache.get();
        if(cache==nullptr)
            return cache;
    }    

    bmcv_convert_to_attr param;
    param.alpha_0=(float)factor.val[0];
    param.alpha_1=(float)factor.val[1];
    param.alpha_2=(float)factor.val[2];
    param.beta_0=(float)value.val[0];
    param.beta_1=(float)value.val[1];
    param.beta_2=(float)value.val[2];
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(input->get_handle());
    result->data.handle=input->data.handle;
    bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);
    result->owned=true;
    result->is_rgb=input->is_rgb;
    if(value.val[0]!=0 && value.val[1]!=0 && value.val[2]!=0 ){
        bm_image temp_image;
        bm_image_create(result->data.handle,get_height(),get_width(),input->data.image.image_format,input->data.image.data_type,&temp_image);
        bm_image_alloc_dev_mem(temp_image);
        bmcv_convert_to_attr param1=param;
        param1.alpha_2=1;
        param1.beta_2=0;
        bmcv_image_convert_to(result->data.handle, 1,param1, &(input->data.image), &temp_image);

        bmcv_convert_to_attr param2=param;
        param2.alpha_0=1;
        param2.alpha_1=1;
        param2.beta_0=0;
        param2.beta_1=0;
        bmcv_image_convert_to(result->data.handle, 1,param2, &temp_image, &(result->data.image));
        bm_image_destroy(temp_image);
    }
    else{
        bmcv_image_convert_to(result->data.handle, 1,param, &(input->data.image), &(result->data.image));
    }
    std::shared_ptr<QyImage> result_out=result;
    return result_out;

};


std::shared_ptr<QyImage> from_data(uint8_t* img_data,int element_size,int src_stride,int width,int height,std::shared_ptr<Device_Handle> handle,bool is_rgb,bool from_host){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(handle);
    std::shared_ptr<Device_Handle_bm> hd=std::dynamic_pointer_cast<Device_Handle_bm>(handle);
    if(hd!=nullptr){
        result->data.handle=hd->handle;
    }
    result->data.handle=hd->handle;
    bm_image_create(result->data.handle,height,width, FORMAT_BGR_PACKED,DATA_TYPE_EXT_1N_BYTE,&(result->data.image));
    bm_image_alloc_dev_mem(result->data.image);


    result->set_is_rgb(is_rgb);
    result->set_owned(true);

    bm_device_mem_t mem;
    bm_image_get_device_mem(result->data.image, &mem);

    if(element_size*width<src_stride){
        for(int i=0;i<height;i++){
            bm_memcpy_s2d_partial_offset(result->data.handle,mem,img_data+i*src_stride,width*element_size,i*width*element_size);
        }
    }   
    else{
        bm_memcpy_s2d(result->data.handle,mem,img_data);
    } 
    std::shared_ptr<QyImage> result_out=result;
    return result_out;
};

std::shared_ptr<QyImage> from_mat(cv::Mat img,std::shared_ptr<Device_Handle> handle,bool is_rgb){
    std::shared_ptr<QyImage_bm> result=std::make_shared<QyImage_bm>(handle);
    std::shared_ptr<Device_Handle_bm> hd=std::dynamic_pointer_cast<Device_Handle_bm>(handle);
    if(hd!=nullptr){
        result->data.handle=hd->handle;
        result->set_image(img,is_rgb);
    }
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

