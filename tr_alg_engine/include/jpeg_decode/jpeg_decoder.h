#ifndef __JPED_DECODER_H__
#define __JPED_DECODER_H__

#include <vector>
#include <iostream>
#include <cv_lib/type_def.h>
#include <network_engine/device_handle.h>
#include <shared_mutex>

class Jpeg_Decoder{
protected:

    std::vector<std::shared_ptr<Device_Handle>> devices_handles;
    int max_wait=0;
    int max_decoder=4;
    int init_cnt=4;
    int cnt=0;
    std::shared_mutex mutex;

public:
    Jpeg_Decoder(int max_wait,int max_decoder){
        this->max_wait=max_wait;
        this->max_decoder=max_decoder;
    };
    virtual ~Jpeg_Decoder(){

    };
    virtual void init(std::vector<std::shared_ptr<Device_Handle>> devices_handles,int init_cnt)=0;

    virtual std::shared_ptr<QyImage> decode(const std::vector<unsigned char>& image)=0;

};


extern "C" std::shared_ptr<Jpeg_Decoder> get_jpeg_decoder(int max_wait=5,int max_decoder=32);

#endif
