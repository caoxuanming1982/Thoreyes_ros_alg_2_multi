#include "jpeg_decode/bm/jpeg_decoder_cv.h"

Jpeg_Decoder_cv::Jpeg_Decoder_cv(int max_wait, int max_decoder) : Jpeg_Decoder(max_wait, max_decoder) {

};
Jpeg_Decoder_cv::~Jpeg_Decoder_cv() {

};
void Jpeg_Decoder_cv::init(std::vector<std::shared_ptr<Device_Handle>> devices_handles, int init_cnt)
{
    this->devices_handles = devices_handles;
    this->init_cnt = init_cnt;
};
std::shared_ptr<QyImage> Jpeg_Decoder_cv::decode(const std::vector<unsigned char> &data)
{
    cv::Mat image = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
    int current_idx;
    {
        std::unique_lock lock(mutex);
        current_idx = cnt;
        cnt += 1;
    }
    std::shared_ptr<QyImage> result = from_mat(image, devices_handles[current_idx % devices_handles.size()]);
    return result;
};

extern "C" std::shared_ptr<Jpeg_Decoder> get_jpeg_decoder(int max_wait,int max_decoder){
    std::shared_ptr<Jpeg_Decoder_cv> decoder=std::make_shared<Jpeg_Decoder_cv>(max_wait,max_decoder);
    std::shared_ptr<Jpeg_Decoder> res=decoder;
    return res;
};
