#include "cv_lib/hw/type_def_hw.h"
#include "inout_type.h"
#include "network_engine/hw/device_handle_hw.h"
int QyImage_hw::get_width()
{

    return this->data.image.picture_width;
};
int QyImage_hw::get_height()
{
    return this->data.image.picture_height;
};
bool QyImage_hw::is_empty()
{
    if (data.image.picture_width <= 0 || data.image.picture_height <= 0 || data.image.picture_address == nullptr)
        return true;
    return false;
};

std::shared_ptr<QyImage> QyImage_hw::copy()
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    result->data.image.picture_width = get_width();
    result->data.image.picture_height = get_height();
    if (is_rgb)
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
    }
    else
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    result->data.image.picture_height_stride = result->data.image.picture_height;
    result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);
    ret = aclrtMemcpy(result->data.image.picture_address, result->data.image.picture_buffer_size, this->data.image.picture_address, this->data.image.picture_buffer_size, aclrtMemcpyKind::ACL_MEMCPY_Device_TO_DEVICE);

    result->is_rgb = this->is_rgb;
    result->data.chn=this->data.chn;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_hw::crop(cv::Rect box)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    result->data.image.picture_width = box.width;
    result->data.image.picture_height = box.height;
    if (is_rgb)
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
    }
    else
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    result->data.image.picture_height_stride = result->data.image.picture_height;
    result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);

    hi_vpc_crop_region_info cropRegionInfos[1];
    cropRegionInfos[0].dest_pic_info = result->data.image;
    cropRegionInfos[0].crop_region.left_offset = box.x;
    cropRegionInfos[0].crop_region.top_offset = box.y;
    cropRegionInfos[0].crop_region.crop_width = box.width;
    cropRegionInfos[0].crop_region.crop_height = box.height;

    uint32_t taskID = 0;
    ret = hi_mpi_vpc_crop(this->data.chn->get_chn(), &this->data.image, cropRegionInfos, 1, &taskID, -1);

    uint32_t taskIDResult = taskID;
    ret = hi_mpi_vpc_get_process_result(this->data.chn->get_chn(), taskIDResult, -1);
    result->data.chn=this->data.chn;

    result->is_rgb = this->is_rgb;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_hw::resize(int width, int height, bool use_bilinear)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    result->data.image.picture_width = width;
    result->data.image.picture_height = height;
    if (is_rgb)
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
    }
    else
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    result->data.image.picture_height_stride = result->data.image.picture_height;
    result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);

    uint32_t taskID = 0;
    if (use_bilinear)
    {

        ret = hi_mpi_vpc_resize(this->data.chn->get_chn(), &this->data.image, &result->data.image, 0, 0, 0, &taskID, -1);
    }
    else
    {
        ret = hi_mpi_vpc_resize(this->data.chn->get_chn(), &this->data.image, &result->data.image, 0, 0, 1, &taskID, -1);
    }

    uint32_t taskIDResult = taskID;
    ret = hi_mpi_vpc_get_process_result(this->data.chn->get_chn(), taskIDResult, -1);

    result->data.chn=this->data.chn;
    result->is_rgb = this->is_rgb;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_hw::crop_resize(cv::Rect box, int width, int height, bool use_bilinear)
{

    if (use_bilinear)
    {
        std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
        result->data.image.picture_width = width;
        result->data.image.picture_height = height;
        if (is_rgb)
        {
            result->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
        }
        else
        {
            result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
        }
        result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
        result->data.image.picture_height_stride = result->data.image.picture_height;
        result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
        int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);

        hi_vpc_crop_region_info cropRegionInfos[1];
        cropRegionInfos[0].dest_pic_info = result->data.image;
        cropRegionInfos[0].crop_region.left_offset = box.x;
        cropRegionInfos[0].crop_region.top_offset = box.y;
        cropRegionInfos[0].crop_region.crop_width = box.width;
        cropRegionInfos[0].crop_region.crop_height = box.height;

        uint32_t taskID = 0;
        hi_mpi_vpc_crop_resize(this->data.chn->get_chn(), &this->data.image, cropRegionInfos, 1, &taskID, -1);
        uint32_t taskIDResult = taskID;
        ret = hi_mpi_vpc_get_process_result(this->data.chn->get_chn(), taskIDResult, -1);
        result->is_rgb = this->is_rgb;
        std::shared_ptr<QyImage> result_out = result;
        return result_out;
    }
    else
    {
        std::shared_ptr<QyImage> temp = this->crop(box);
        temp = temp->resize(width, height, false);
        std::shared_ptr<QyImage> result_out = temp;
        return result_out;
    }
};

std::shared_ptr<QyImage> QyImage_hw::padding(int left, int right, int up, int down, int value)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    int width=this->get_width() + left + right;
    int height=this->get_height() + up + down;


    hi_vpc_make_border_info border_info;
    border_info.left=left;
    border_info.right=right;
    border_info.top=up;
    border_info.bottom=down;

    border_info.border_type=HI_BORDER_CONSTANT;
    border_info.scalar_value.val[0]=value;
    border_info.scalar_value.val[1]=value;
    border_info.scalar_value.val[2]=value;
    border_info.scalar_value.val[3]=value;

    result->data.image.picture_width = width;
    result->data.image.picture_height = height;
    if (is_rgb)
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
    }
    else
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    result->data.image.picture_height_stride = result->data.image.picture_height;
    result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);

    uint32_t taskID = 0;
    ret=hi_mpi_vpc_copy_make_border(this->data.chn->get_chn(),&this->data.image, &result->data.image,border_info,&taskID, -1);
    uint32_t taskIDResult = taskID;
    ret = hi_mpi_vpc_get_process_result(this->data.chn->get_chn(), taskIDResult, -1);


    result->data.chn=this->data.chn;
    result->is_rgb = this->is_rgb;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_hw::warp_affine(std::vector<float> &matrix, int width, int height, bool use_bilinear)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    cv::Mat input=this->get_image();
    cv::Mat M(2,3,CV_32FC1);
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            M.at<float>(i,j)=matrix[i*3+j];            
        }
    }
    cv::Mat output;

    if(use_bilinear){
        cv::warpAffine(input,output,M,cv::Size(width,height),cv::INTER_LINEAR,cv::BORDER_REPLICATE);

    }
    else{
        cv::warpAffine(input,output,M,cv::Size(width,height),cv::INTER_NEAREST,cv::BORDER_REPLICATE);
    }
    result->set_image(output,is_rgb);

    result->is_rgb = this->is_rgb;
    result->data.chn=this->data.chn;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_hw::warp_perspective(std::vector<float> &matrix, int width, int height, bool use_bilinear)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    cv::Mat input=this->get_image();

    cv::Mat M(3, 3, CV_32FC1);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            M.at<float>(i, j) = matrix[i * 3 + j];
        }
    }

    cv::Mat output;

    if(use_bilinear){
        cv::warpPerspective(input,output,M,cv::Size(width,height),cv::INTER_LINEAR);
    }
    else{
        cv::warpPerspective(input,output,M,cv::Size(width,height),cv::INTER_NEAREST);
    }
    result->set_image(output,is_rgb);
    result->is_rgb = this->is_rgb;
    result->data.chn=this->data.chn;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};
std::shared_ptr<QyImage> QyImage_hw::warp_perspective(std::vector<cv::Point2f> &points, int width, int height, bool use_bilinear)
{
    std::vector<cv::Point2f> coordinate_dst(4);
    coordinate_dst[0].x = 0;
    coordinate_dst[0].y = 0;
    coordinate_dst[1].x = width - 1;
    coordinate_dst[1].y = 0;
    coordinate_dst[2].x = 0;
    coordinate_dst[2].y = height - 1;
    coordinate_dst[3].x = width - 1;
    coordinate_dst[3].y = height - 1;
    cv::Mat M = cv::getPerspectiveTransform(points, coordinate_dst);
    M.convertTo(M, CV_32F);
    std::vector<float> M_v;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            M_v.push_back(M.at<float>(i, j));
        }
    }
    return warp_perspective(M_v, width, height, use_bilinear);
};

std::vector<std::shared_ptr<QyImage>> QyImage_hw::batch_warp_affine(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false){
    std::vector<std::shared_ptr<QyImage>> result_all;
    cv::Mat input=this->get_image();
    for(int k=0;k<matrixes_in.size();k++){
        cv::Mat M(2,3,CV_32FC1);
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                M.at<float>(i,j)=matrixes_in[k][i*3+j];            
            }
        }
        cv::Mat output;
        if(use_bilinear){
            cv::warpAffine(input,output,M,cv::Size(width,height),cv::INTER_LINEAR,cv::BORDER_REPLICATE);

        }
        else{
            cv::warpAffine(input,output,M,cv::Size(width,height),cv::INTER_NEAREST,cv::BORDER_REPLICATE);
        }

        std::shared_ptr<QyImage_hw> result=std::make_shared<QyImage_hw>(this->get_handle());
        result->set_image(output,is_rgb);
        result->is_rgb = this->is_rgb;
        result->data.chn=this->data.chn;
        std::shared_ptr<QyImage> result_out = result;
        result_all.push_back(result_out);    
    }
    return result_all;
}

std::vector<std::shared_ptr<QyImage>> QyImage_hw::batch_warp_perspective(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false){
    std::vector<std::shared_ptr<QyImage>> result_all;
    cv::Mat input=this->get_image();
    for(int k=0;k<matrixes_in.size();k++){
        cv::Mat M(2,3,CV_32FC1);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                M.at<float>(i, j) = matrixes_in[k][i * 3 + j];
            }
        }

        cv::Mat output;
        if(use_bilinear){
            cv::warpPerspective(input,output,M,cv::Size(width,height),cv::INTER_LINEAR,cv::BORDER_REPLICATE);

        }
        else{
            cv::warpPerspective(input,output,M,cv::Size(width,height),cv::INTER_NEAREST,cv::BORDER_REPLICATE);
        }

        std::shared_ptr<QyImage_hw> result=std::make_shared<QyImage_hw>(this->get_handle());
        result->set_image(output,is_rgb);
        result->is_rgb = this->is_rgb;
        result->data.chn=this->data.chn;
        std::shared_ptr<QyImage> result_out = result;
        result_all.push_back(result_out);    
    }
    return result_all;

}
std::vector<std::shared_ptr<QyImage>> QyImage_hw::batch_warp_perspective(std::vector<std::vector<cv::Point2f>>& pointss,int width,int height,bool use_bilinear=false){

    std::vector<std::shared_ptr<QyImage>> result_all;
    cv::Mat input=this->get_image();
    for(int k=0;k<pointss.size();k++){

        std::vector<cv::Point2f> coordinate_dst(4);
        coordinate_dst[0].x = 0;
        coordinate_dst[0].y = 0;
        coordinate_dst[1].x = width - 1;
        coordinate_dst[1].y = 0;
        coordinate_dst[2].x = 0;
        coordinate_dst[2].y = height - 1;
        coordinate_dst[3].x = width - 1;
        coordinate_dst[3].y = height - 1;
        cv::Mat M = cv::getPerspectiveTransform(pointss[k], coordinate_dst);

        cv::Mat output;
        if(use_bilinear){
            cv::warpPerspective(input,output,M,cv::Size(width,height),cv::INTER_LINEAR,cv::BORDER_REPLICATE);

        }
        else{
            cv::warpPerspective(input,output,M,cv::Size(width,height),cv::INTER_NEAREST,cv::BORDER_REPLICATE);
        }

        std::shared_ptr<QyImage_hw> result=std::make_shared<QyImage_hw>(this->get_handle());
        result->set_image(output,is_rgb);
        result->is_rgb = this->is_rgb;
        result->data.chn=this->data.chn;
        std::shared_ptr<QyImage> result_out = result;
        result_all.push_back(result_out);    
    }
    return result_all;

}

std::shared_ptr<QyImage> QyImage_hw::cvtcolor(bool to_rgb)
{
    if (to_rgb == this->is_rgb)
    {
        return copy();
    }

    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(this->get_handle());
    result->data.image.picture_width = get_width();
    result->data.image.picture_height = get_height();
    if (is_rgb)
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    else
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    result->data.image.picture_height_stride = result->data.image.picture_height;
    result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);

    uint32_t taskID = 0;
    ret = hi_mpi_vpc_convert_color(this->data.chn->get_chn(), &this->data.image, &result->data.image, &taskID, -1);

    uint32_t taskIDResult = taskID;
    ret = hi_mpi_vpc_get_process_result(this->data.chn->get_chn(), taskIDResult, -1);

    result->data.chn=this->data.chn;
    result->is_rgb = !this->is_rgb;
    std::shared_ptr<QyImage> result_out = result;
    return result_out;
};

std::shared_ptr<QyImage> QyImage_hw::convertTo(QyImage::Data_type t)
{
    std::shared_ptr<QyImage_hw> result;
    hi_pixel_format dst_fmt;
    int stride;
    switch (t)
    {

    case QyImage::Data_type::UInt8:
        if (this->is_rgb)
        {
            dst_fmt = hi_pixel_format::HI_PIXEL_FORMAT_RGB_888;
        }
        else
        {
            dst_fmt = hi_pixel_format::HI_PIXEL_FORMAT_BGR_888;
        }
        break;
    default:
        dst_fmt = hi_pixel_format::HI_PIXEL_FORMAT_UNKNOWN;
        break;
    }
    if (dst_fmt == hi_pixel_format::HI_PIXEL_FORMAT_UNKNOWN)
    {
        return result;
    }
    return this->copy();
};

cv::Mat QyImage_hw::get_image()
{
    cv::Mat result(this->get_height(), this->get_width(), CV_8UC3);
    int ret = aclrtMemcpy2d(result.data, this->get_width() * 3, this->data.image.picture_address, this->data.image.picture_width_stride, this->data.image.picture_width, this->data.image.picture_height, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST);
    return result;
};

void QyImage_hw::set_image(cv::Mat input, bool is_rgb)
{

    if (this->data.image.picture_address != nullptr)
    {
        hi_mpi_dvpp_free(0, this->data.image.picture_address);
        this->data.image.picture_address = nullptr;
    }

    this->data.image.picture_width = input.cols;
    this->data.image.picture_height = input.rows;
    if (is_rgb)
    {
        this->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
    }
    else
    {
        this->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    this->data.image.picture_width_stride = (int)((this->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    this->data.image.picture_height_stride = this->data.image.picture_height;
    this->data.image.picture_buffer_size = this->data.image.picture_width_stride * this->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &this->data.image.picture_address, this->data.image.picture_buffer_size);

    ret = aclrtMemcpy2d(this->data.image.picture_address, this->data.image.picture_width_stride, input.data, input.step, this->data.image.picture_width, this->data.image.picture_height, aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);
    this->is_rgb = is_rgb;
};

std::shared_ptr<QyImage> from_data(uint8_t *img_data, int element_size, int src_stride, int width, int height, std::shared_ptr<Device_Handle> handle, bool is_rgb, bool from_host)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(handle);
    if (result->data.image.picture_address != nullptr)
    {
        hi_mpi_dvpp_free(0, result->data.image.picture_address);
        result->data.image.picture_address = nullptr;
    }

    result->data.image.picture_width = width;
    result->data.image.picture_height = height;
    if (is_rgb)
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_RGB_888;
    }
    else
    {
        result->data.image.picture_format == HI_PIXEL_FORMAT_BGR_888;
    }
    result->data.image.picture_width_stride = (int)((result->data.image.picture_width * 3 + 16 - 1) / 16) * 16;
    result->data.image.picture_height_stride = result->data.image.picture_height;
    result->data.image.picture_buffer_size = result->data.image.picture_width_stride * result->data.image.picture_height_stride * 3 / 2;
    int32_t ret = hi_mpi_dvpp_malloc(0, &result->data.image.picture_address, result->data.image.picture_buffer_size);

    if (from_host)
    {
        ret = aclrtMemcpy2d(result->data.image.picture_address, result->data.image.picture_width_stride, img_data, src_stride, result->data.image.picture_width, result->data.image.picture_height, aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);
    }
    else
    {
        ret = aclrtMemcpy2d(result->data.image.picture_address, result->data.image.picture_width_stride, img_data, src_stride, result->data.image.picture_width, result->data.image.picture_height, aclrtMemcpyKind::ACL_MEMCPY_Device_TO_DEVICE);
    }
    result->data.chn->init();
    result->set_is_rgb(is_rgb);
    result->set_owned(true);
    return result;
};

std::shared_ptr<QyImage> from_mat(cv::Mat img, std::shared_ptr<Device_Handle> handle, bool is_rgb)
{
    std::shared_ptr<QyImage_hw> result = std::make_shared<QyImage_hw>(handle);
    result->set_image(img, is_rgb);
    result->data.chn = std::make_shared<HW_DVPP_Chn>();
    result->data.chn->init();
    return result;
};
std::shared_ptr<QyImage> from_InputOutput(std::shared_ptr<InputOutput> src, bool copy)
{
    if (src->data_type != InputOutput::Type::Image_t)
        return std::shared_ptr<QyImage>();
    if (copy)
    {
        return src->data.image->copy();
    }
    else
    {
        return src->data.image;
    }
};
