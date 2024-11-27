#ifndef __JPED_DECODER_IX_H__
#define __JPED_DECODER_IX_H__

#include <vector>
#include <iostream>
#include <cv_lib/type_def.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <thread>

#include "jpeg_decode/jpeg_decoder.h"
#include "common.h"
#include "help_func.cuh"

class GpuMat{
public:
    int cols=0;
    int rows=0;
    int step=0;
    int element_size=0;
    int padding_to=64;
    int deep=0;
    int channel=0;
    uint8_t*data=nullptr;

    GpuMat(){

    };

    uint8_t* cudaPtr(){
        return data;
    };

    void from_channels(GpuMat&c1,GpuMat&c2,GpuMat&c3){
        int channel=0;
        if(c1.data!=nullptr){
            channel+=1;
        }
        if(c2.data!=nullptr){
            channel+=1;
        }
        if(c3.data!=nullptr){
            channel+=1;
        }
        if(cols!=c1.cols||rows!=c1.rows){
            re_init(c1.rows,c1.cols,deep,channel);
        }
        
        merge_func_8u(data,this->channel,step,c1.data,c2.data,c3.data,c1.step,cols,rows,NULL);
    };

    GpuMat(int height,int width,int type,int channel){
        deep = type & CV_MAT_DEPTH_MASK;
        switch(deep){
            case CV_8U:
                element_size=channel*1;
                break;
            case CV_32F:
                element_size=channel*4;
                break;
            default:
                element_size=channel*1;
                break;

        }
        this->channel=channel;
        step=((width*element_size+padding_to-1)/padding_to)*padding_to;

        cudaMalloc((void**)&(data),height*step);
        cudaMemset(data,0,height*step);
        rows=height;
        cols=width;
    };
    void re_init(int height,int width,int type,int channel){
        if(data!=nullptr){
            cudaFree(data);
            data=nullptr;

        }

        deep = type & CV_MAT_DEPTH_MASK;
        this->channel=channel;
        switch(deep){
            case CV_8U:
                element_size=channel*1;
                break;
            case CV_32F:
                element_size=channel*4;
                break;
            default:
                element_size=channel*1;
                break;

        }
        step=((width*element_size+padding_to-1)/padding_to)*padding_to;

        cudaMalloc((void**)&(data),height*step);
        cudaMemset(data,0,height*step);
        rows=height;
        cols=width;

    };
    ~GpuMat(){
        cudaFree(data);
        data=nullptr;
    };
};
class CudaJpegDecode
{
public:

    std::shared_ptr<Device_Handle> device;

    explicit CudaJpegDecode();
    ~CudaJpegDecode();

    bool DeviceInit(const nvjpegOutputFormat_t out_fmt, std::shared_ptr<Device_Handle> device);
    bool DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, std::shared_ptr<Device_Handle> device);
    
    bool Decode(const uchar *image, const int length, GpuMat& dst, bool pipelined = false);

private:
    static int host_malloc(void **p, size_t s, unsigned int f);
    static int host_free(void *p);
    static int dev_malloc(void **p, size_t s);
    static int dev_free(void *p);
    int ConvertSMVer2Cores(int major, int minor);
    int GpuGetMaxGflopsDeviceId();
    bool inited=false;

//    int device_id_ = 0;
    nvjpegDevAllocator_t dev_allocator_;
    nvjpegPinnedAllocator_t pinned_allocator_;
    nvjpegHandle_t nvjpeg_handle_;
    nvjpegJpegState_t nvjepg_state_;
    nvjpegOutputFormat_t out_fmt_;
    int batch_size_ = 0;
    
    // used with decoupled API
    nvjpegJpegDecoder_t nvjpeg_decoder_;
    nvjpegJpegState_t nvjpeg_decoupled_state_;
    nvjpegBufferPinned_t pinned_buffers_;
    nvjpegBufferDevice_t device_buffer_;
    nvjpegJpegStream_t jpeg_streams_;
    nvjpegDecodeParams_t nvjpeg_decode_params_;
};


class Jpeg_Decoder_ix:public Jpeg_Decoder{
    std::vector<CudaJpegDecode> decoders;
    std::vector<int> decoder_states;
    nvjpegOutputFormat_t format;

    int current_cnt=0;
public:
    Jpeg_Decoder_ix(int max_wait=5,int max_decoder=32);
    virtual ~Jpeg_Decoder_ix();
    virtual void init(std::vector<std::shared_ptr<Device_Handle>> devices_handles,int init_cnt=4);
    virtual std::shared_ptr<QyImage> decode(const std::vector<unsigned char>& data);
    int add_instance(std::shared_ptr<Device_Handle> device);
    int get_instance_idx();

};

#endif
