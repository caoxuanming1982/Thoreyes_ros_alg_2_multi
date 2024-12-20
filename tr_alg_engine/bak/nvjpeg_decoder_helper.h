#ifndef __NVJPED_DECODER_HELPER__
#define __NVJPED_DECODER_HELPER__
#ifndef USE_BM
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <shared_mutex>
class CudaJpegDecode
{
public:
    explicit CudaJpegDecode();
    ~CudaJpegDecode();

    bool DeviceInit(const nvjpegOutputFormat_t out_fmt, const int device = 0);

    /**
     * @description: 初始化硬件及nvjpeg
     * @param {batch_size} 批处理时，批次大小；不足一批的从list开头继续补充
     * @param {max_cpu_threads} cpu的最大线程数
     * @param {out_fmt} 解出的格式
     * @return {bool} 成功true, 失败false
     */    
    bool DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, const int device = 0);
    
    /**
     * @description: 解压缩一张图片
     * @param {image} 输入图片的内容地址
     * @param {length} 输入图片的内容长度
     * @param {pipelined} true使用解耦的接口形式
     * @return {dst} 解压缩的图片，　cv::Mat或者cv::cuda::Mat
     * @return {bool} 成功true, 失败false
     */  
    bool Decode(uchar *image, const int length, cv::OutputArray dst, bool pipelined = false);

    /**
     * @description: 解压缩批量图片
     * @param {images} 所有图片的数据的每个图片地址，个数需要小于等于DeviceInit中的batch_size 
     * @param {lengths} 所有图片的数据的每个图片长度，与images顺序和长度一致
     * @return {dst} 解出的图上，可以是std::vector<cv::Mat>或者std::vector<cv::cuda::GpuMat>
     * @return {bool} 解压缩成功true, 失败false
     */  
    bool Decode(const std::vector<uchar*> &images, const std::vector<size_t> lengths, cv::OutputArray &dst);

private:
    static int host_malloc(void **p, size_t s, unsigned int f);
    static int host_free(void *p);
    static int dev_malloc(void **p, size_t s);
    static int dev_free(void *p);
    int ConvertSMVer2Cores(int major, int minor);
    int GpuGetMaxGflopsDeviceId();
    bool inited=false;

    int device_id_ = 0;
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


class CudaJpegDecode_pool{
    std::vector<CudaJpegDecode> decoders;
    std::vector<int> decoder_states;
    int max_wait=5;
    int current_cnt=0;
    int max_cnt=32;
    nvjpegOutputFormat_t format;
    std::shared_mutex mutex;
    std::vector<int> devices_ids;

public:
    CudaJpegDecode_pool();
    CudaJpegDecode_pool(int max_wait,int max_decoder);

    bool init(nvjpegOutputFormat_t format,std::vector<int> devices_ids,int init_cnt=4);

    bool decode(uchar *image, const int length, cv::OutputArray dst);
    int add_instance(int device_id=-1);
    int get_instance_idx();



};
#endif
#endif