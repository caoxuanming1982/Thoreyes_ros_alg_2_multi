#ifndef __JPED_DECODER_HW_H__
#define __JPED_DECODER_HW_H__

#include <vector>
#include <iostream>
#include <cv_lib/type_def.h>
#include <cv_lib/hw/type_def_hw.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include "jpeg_decode/jpeg_decoder.h"
#include <network_engine/hw/device_handle_hw.h>
#include "common.h"
#include <thread>

class Buffer{
    
public:
    uint8_t* data=nullptr;
    int size=0;
    int cap=0;
    Buffer(int size_in=0){
        if(size_in!=0){
            acldvppMalloc(&data,size_in);
            size=size_in;
            cap=size_in;

        }
    }
    ~Buffer(){
        if( data!=nullptr)
        {            
            acldvppFree(data);
        }
    }

    void resize(int new_size){
        if(new_size>cap && data!=nullptr)
        {            
            acldvppFree(data);
            acldvppMalloc(&data,new_size);
            size=new_size;
            cap=new_size;
        }
        else{
            size=new_size;
        }
    }

};

class Jped_Decoder_hw_impl{
    std::shared_ptr<Device_Handle> handle;
    acldvppChannelDesc dvppChannelDesc_;
    Buffer input_buffer;
    Buffer output_buffer;
    bool inited=false;
public:
    bool init(std::shared_ptr<Device_Handle> handle){
        if(inited==false){
            dvppChannelDesc_ = acldvppCreateChannelDesc();
            aclError ret = acldvppCreateChannel(dvppChannelDesc_);            
            this->handle=handle;
        }

    };

    ~Jped_Decoder_hw_impl(){
        if(inited){
            acldvppDestroyChannel(dvppChannelDesc_);
            (void)acldvppDestroyChannelDesc(dvppChannelDesc_);
            dvppChannelDesc_ = nullptr;
        }
    }

    bool decode(const std::vector<unsigned char>& data,std::shared_ptr<QyImage>& output){
        uint32_t width;
        uint32_t height;
        int32_t components;
        acldvppJpegFormat format;
        ret = acldvppJpegGetImageInfoV2(data.data(), data.size(),&width,&height,&components,&format);
        if(format!=acldvppPixelFormat::PIXEL_FORMAT_BGR_888){
            return false;
        }

        uint32_t decodeOutBufferSize = 0;
        ret = acldvppJpegPredictDecSize(data.data(), data.size(), PIXEL_FORMAT_BGR_888,&decodeOutBufferSize);

        input_buffer.resize(data.size());
        output_buffer.resize(decodeOutBufferSize);
        aclrtMemcpy(input_buffer.data, input_buffer.size, data.data(), data.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        acldvppPicDesc decodeOutputDesc_ = acldvppCreatePicDesc();
        acldvppSetPicDescData(decodeOutputDesc_, output_buffer.data);
        acldvppSetPicDescFormat(decodeOutputDesc_, PIXEL_FORMAT_BGR_888); 
        acldvppSetPicDescWidth(decodeOutputDesc_, width);
        acldvppSetPicDescHeight(decodeOutputDesc_, height);
        acldvppSetPicDescWidthStride(decodeOutputDesc_, (int)((width * 3 + 16 - 1) / 16) * 16);
        acldvppSetPicDescHeightStride(decodeOutputDesc_, height);
        acldvppSetPicDescSize(decodeOutputDesc_, decodeOutBufferSize);        

        std::shared_ptr<Device_Handle_hw>hd= std::dynamic_pointer_cast<Device_Handle_hw>(handle);
        ret = acldvppJpegDecodeAsync(dvppChannelDesc_, input_buffer.data, input_buffer.size, decodeOutputDesc_, hd->handle);
        ret = aclrtSynchronizeStream(hd->handle);


        output=from_data(output_buffer.data,3,(int)((width * 3 + 16 - 1) / 16) * 16,width,height,handle,false,false);
        acldvppDestroyPicDesc(decodeOutputDesc_);
    };

};


class Jpeg_Decoder_hw:public Jpeg_Decoder{
public:

    std::vector<Jped_Decoder_hw_impl> decoders;
    std::vector<int> decoder_states;

    int current_cnt=0;


    Jpeg_Decoder_hw(int max_wait=5,int max_decoder=32);
    virtual ~Jpeg_Decoder_hw();
    virtual void init(std::vector<std::shared_ptr<Device_Handle>> devices_handles,int init_cnt=4);
    virtual std::shared_ptr<QyImage> decode(const std::vector<unsigned char>& data);
    int add_instance(std::shared_ptr<Device_Handle> device);
    int get_instance_idx();
};

#endif
