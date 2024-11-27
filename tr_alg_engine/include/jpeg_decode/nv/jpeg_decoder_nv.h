#ifndef __JPED_DECODER_NV_H__
#define __JPED_DECODER_NV_H__

#include <vector>
#include <iostream>
#include <cv_lib/type_def.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <thread>

#include <opencv2/imgcodecs/legacy/constants_c.h>
#include "jpeg_decode/jpeg_decoder.h"
#include "common.h"

class CudaJpegDecode
{
public:

    std::shared_ptr<Device_Handle> device;

    explicit CudaJpegDecode();
    ~CudaJpegDecode();

    bool DeviceInit(const nvjpegOutputFormat_t out_fmt, std::shared_ptr<Device_Handle> device);
    bool DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, std::shared_ptr<Device_Handle> device);
    
    bool Decode(const uchar *image, const int length, cv::OutputArray dst, bool pipelined = false);
    bool Decode(const std::vector<const uchar*> &images, const std::vector<size_t> lengths, cv::OutputArray &dst);

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


class Jpeg_Decoder_nv:public Jpeg_Decoder{
    std::vector<CudaJpegDecode> decoders;
    std::vector<int> decoder_states;
    nvjpegOutputFormat_t format;

    int current_cnt=0;
public:
    Jpeg_Decoder_nv(int max_wait=5,int max_decoder=32);
    virtual ~Jpeg_Decoder_nv();

    virtual void init(std::vector<std::shared_ptr<Device_Handle>> devices_handles,int init_cnt=4);
    virtual std::shared_ptr<QyImage> decode(const std::vector<unsigned char>& data);
    int add_instance(std::shared_ptr<Device_Handle> device);
    int get_instance_idx();

};

#endif
