#ifndef __JPED_DECODER_CV_H__
#define __JPED_DECODER_CV_H__

#include <vector>
#include <iostream>
#include <cv_lib/type_def.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include "jpeg_decode/jpeg_decoder.h"

class Jpeg_Decoder_cv:public Jpeg_Decoder{
public:
    Jpeg_Decoder_cv(int max_wait=5,int max_decoder=32);
    virtual ~Jpeg_Decoder_cv();
    virtual void init(std::vector<std::shared_ptr<Device_Handle>> devices_handles,int init_cnt=4);
    virtual std::shared_ptr<QyImage> decode(const std::vector<unsigned char>& data);

};

#endif
