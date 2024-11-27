#include "cv_lib/type_def.h"


QyImage::QyImage(std::shared_ptr<Device_Handle> handle):handle(handle){

};
QyImage::~QyImage(){

};
void QyImage::set_is_rgb(bool is_rgb){
    this->is_rgb=is_rgb;
};

bool QyImage::get_is_rgb(){
    return this->is_rgb;
}
std::shared_ptr<Device_Handle> QyImage::get_handle(){
    return this->handle;
};

void QyImage::set_owned(bool owned){
    this->owned=owned;
};


std::shared_ptr<QyImage> QyImage::resize_keep_ratio(int width, int height, int value, Padding_mode mode, bool use_bilinear)
{
    float factor_w = 1.0f * width / this->get_width();
    float factor_h = 1.0f * height / this->get_height();
    float factor = std::min(factor_w, factor_h);
    int target_width = int(this->get_width() * factor);
    int target_height = int(this->get_height() * factor);

    std::shared_ptr<QyImage> resize_cache = this->resize(target_width, target_height, use_bilinear);
    std::shared_ptr<QyImage> result = resize_cache->padding_to(width, height, value, mode);
    return result;
};
std::shared_ptr<QyImage> QyImage::crop_resize_keep_ratio(cv::Rect box, int width, int height, int value, Padding_mode mode, bool use_bilinear)
{
    int current_w=box.width;
    int current_h=box.height;

    float factor_w = 1.0f * width / current_w;
    float factor_h = 1.0f * height / current_h;
    float factor = std::min(factor_w, factor_h);
    int target_width = int(current_w * factor);
    int target_height = int(current_h * factor);
    std::shared_ptr<QyImage> resize_cache = this->crop_resize(box, target_width, target_height, use_bilinear);
    std::shared_ptr<QyImage> result = resize_cache->padding_to(width, height, value, mode);
    return result;
};
std::shared_ptr<QyImage> QyImage::padding_to(int width, int height, int value, Padding_mode mode)
{
    int up = 0, down = 0, left = 0, right = 0;
    if (mode == QyImage::Padding_mode::LeftTop)
    {
        down = height - this->get_height();
        right = width - this->get_width();
    }
    else if (mode == QyImage::Padding_mode::Center)
    {
        up = (height - this->get_height()) / 2;
        left = (width - this->get_width()) / 2;
        down = height - this->get_height() - up;
        right = width - this->get_width() - left;
    }
    else
    {
    }
    std::shared_ptr<QyImage> result = this->padding(left, right, up, down, value);
    return result;
};
std::shared_ptr<QyImage> QyImage::warp_affine(cv::Mat &matrix, int width, int height, bool use_bilinear)
{
    cv::Mat temp;
    matrix.convertTo(temp,CV_32FC1);
    std::vector<float> param;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            param.push_back(temp.at<float>(i, j));
        }
    }
    return warp_affine(param, width, height, use_bilinear);
};
std::shared_ptr<QyImage> QyImage::warp_perspective(cv::Mat& matrix,int width,int height,bool use_bilinear){
    cv::Mat temp;
    matrix.convertTo(temp,CV_32FC1);
    std::vector<float> param;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            param.push_back(temp.at<float>(i, j));
        }
    }
    return warp_perspective(param, width, height, use_bilinear);
    
};

std::vector<std::shared_ptr<QyImage>> QyImage::batch_warp_affine(std::vector<std::vector<float>>& matrixes_in, int width, int height, bool use_bilinear)
{
    std::vector<std::shared_ptr<QyImage>> results;
    for (int i = 0; i < matrixes_in.size(); i++)
    {
        results.push_back(this->warp_affine(matrixes_in[i], width, height, use_bilinear));
    }
    return results;
};


std::vector<std::shared_ptr<QyImage>> QyImage::batch_warp_perspective(std::vector<std::vector<float>> &matrixes_in, int width, int height, bool use_bilinear)
{
    std::vector<std::shared_ptr<QyImage>> results;
    for (int i = 0; i < matrixes_in.size(); i++)
    {
        results.push_back(this->warp_perspective(matrixes_in[i], width, height, use_bilinear));
    }
    return results;
};
std::vector<std::shared_ptr<QyImage>> QyImage::batch_warp_perspective(std::vector<std::vector<cv::Point2f>> &pointss, int width, int height, bool use_bilinear)
{
    std::vector<std::shared_ptr<QyImage>> results;
    for (int i = 0; i < pointss.size(); i++)
    {
        results.push_back(this->warp_perspective(pointss[i], width, height, use_bilinear));
    }
    return results;
};
