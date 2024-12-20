#include "jpeg_decode/nv/jpeg_decoder_nv.h"

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName((cudaError_t)result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

CudaJpegDecode::CudaJpegDecode()
{
}

CudaJpegDecode::~CudaJpegDecode()
{
    if (inited)
    {
        checkCudaErrors(cudaSetDevice(device->get_device_id()));
        checkCudaErrors(nvjpegDecodeParamsDestroy(nvjpeg_decode_params_));
        checkCudaErrors(nvjpegJpegStreamDestroy(jpeg_streams_));
        checkCudaErrors(nvjpegBufferPinnedDestroy(pinned_buffers_));
        checkCudaErrors(nvjpegBufferDeviceDestroy(device_buffer_));
        checkCudaErrors(nvjpegJpegStateDestroy(nvjpeg_decoupled_state_));
        checkCudaErrors(nvjpegDecoderDestroy(nvjpeg_decoder_));

        checkCudaErrors(nvjpegJpegStateDestroy(nvjepg_state_));
        checkCudaErrors(nvjpegDestroy(nvjpeg_handle_));
    }
}
bool CudaJpegDecode::DeviceInit(const nvjpegOutputFormat_t out_fmt, std::shared_ptr<Device_Handle> device)
{
    if (device == nullptr)
    {
        this->device = std::shared_ptr<Device_Handle>(get_device_handle( GpuGetMaxGflopsDeviceId()));
        device=this->device;
    }
    else
    {
        this->device = device;
    }
    checkCudaErrors(cudaSetDevice(device->get_device_id()));
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, device->get_device_id()));
    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
           device->get_device_id(), props.name, props.multiProcessorCount,
           props.maxThreadsPerMultiProcessor, props.major, props.minor,
           props.ECCEnabled ? "on" : "off");

    dev_allocator_ = {&CudaJpegDecode::dev_malloc, &CudaJpegDecode::dev_free};
    pinned_allocator_ = {&CudaJpegDecode::host_malloc, &CudaJpegDecode::host_free};
    int flags = 0;
    checkCudaErrors(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator_, &pinned_allocator_, flags, &nvjpeg_handle_));
    checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjepg_state_));
    out_fmt_ = out_fmt;
    batch_size_ = 1;
    checkCudaErrors(nvjpegDecodeBatchedInitialize(nvjpeg_handle_, nvjepg_state_, 1, 1, out_fmt));

    // for pipelined
    checkCudaErrors(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder_));
    checkCudaErrors(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decoupled_state_));

    checkCudaErrors(nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_));
    checkCudaErrors(nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &device_buffer_));

    checkCudaErrors(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_));

    checkCudaErrors(nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));

    inited = true;
    return true;
};

bool CudaJpegDecode::DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, std::shared_ptr<Device_Handle> device)
{
    if (device == nullptr)
    {
        this->device = std::shared_ptr<Device_Handle>(get_device_handle( GpuGetMaxGflopsDeviceId()));
        device=this->device;
    }
    else
    {
        this->device = device;
    }

    checkCudaErrors(cudaSetDevice(device->get_device_id()));
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, device->get_device_id()));
    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
           device->get_device_id(), props.name, props.multiProcessorCount,
           props.maxThreadsPerMultiProcessor, props.major, props.minor,
           props.ECCEnabled ? "on" : "off");

    dev_allocator_ = {&CudaJpegDecode::dev_malloc, &CudaJpegDecode::dev_free};
    pinned_allocator_ = {&CudaJpegDecode::host_malloc, &CudaJpegDecode::host_free};
    int flags = 0;
    checkCudaErrors(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator_, &pinned_allocator_, flags, &nvjpeg_handle_));
    checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjepg_state_));
    out_fmt_ = out_fmt;
    batch_size_ = batch_size;
    checkCudaErrors(nvjpegDecodeBatchedInitialize(nvjpeg_handle_, nvjepg_state_, batch_size, max_cpu_threads, out_fmt));

    // for pipelined
    checkCudaErrors(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder_));
    checkCudaErrors(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decoupled_state_));

    checkCudaErrors(nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_));
    checkCudaErrors(nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &device_buffer_));

    checkCudaErrors(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_));

    checkCudaErrors(nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));
    inited = true;
    return true;
}

bool CudaJpegDecode::Decode(const uchar *image, const int length, cv::OutputArray dst, bool pipelined)
{
    checkCudaErrors(cudaSetDevice(device->get_device_id()));
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvjpegImage_t iout;
    for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
    {
        iout.channel[i] = nullptr;
        iout.pitch[i] = 0;
    }

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;

    checkCudaErrors(nvjpegGetImageInfo(nvjpeg_handle_, image, length, &channels, &subsampling, widths, heights));
    int mul = 1;
    // in the case of interleaved RGB output, write only to single channel, but
    // 3 samples at once
    if (out_fmt_ == NVJPEG_OUTPUT_RGBI || out_fmt_ == NVJPEG_OUTPUT_BGRI)
    {
        channels = 1;
        mul = 3;
    }
    // in the case of rgb create 3 buffers with sizes of original image
    else if (out_fmt_ == NVJPEG_OUTPUT_RGB ||
             out_fmt_ == NVJPEG_OUTPUT_BGR)
    {
        channels = 3;
        widths[1] = widths[2] = widths[0];
        heights[1] = heights[2] = heights[0];
    }

    // prepare output buffer
    cv::cuda::GpuMat c1(heights[0], widths[0], CV_8UC1), c2(heights[0], widths[0], CV_8UC1), c3(heights[0], widths[0], CV_8UC1);
    iout.channel[0] = (unsigned char *)c1.cudaPtr();
    iout.pitch[0] = c1.step;
    iout.channel[1] = (unsigned char *)c2.cudaPtr();
    iout.pitch[1] = c2.step;
    iout.channel[2] = (unsigned char *)c3.cudaPtr();
    iout.pitch[2] = c3.step;

    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
    checkCudaErrors(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    if (!pipelined)
    {
        checkCudaErrors(cudaEventRecord(startEvent, stream));
        checkCudaErrors(nvjpegDecode(nvjpeg_handle_, nvjepg_state_, image, length, out_fmt_, &iout, stream));
        checkCudaErrors(cudaEventRecord(stopEvent, stream));
    }
    else
    {
        checkCudaErrors(cudaEventRecord(startEvent, stream));
        checkCudaErrors(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state_, device_buffer_));
        int buffer_index = 0;
        checkCudaErrors(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params_, out_fmt_));
        checkCudaErrors(nvjpegJpegStreamParse(nvjpeg_handle_, image, length, 0, 0, jpeg_streams_));
        checkCudaErrors(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state_, pinned_buffers_));
        checkCudaErrors(nvjpegDecodeJpegHost(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_, nvjpeg_decode_params_, jpeg_streams_));
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_, jpeg_streams_, stream));
        checkCudaErrors(nvjpegDecodeJpegDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_, &iout, stream));
        checkCudaErrors(cudaEventRecord(stopEvent, stream));
    }
    checkCudaErrors(cudaEventSynchronize(stopEvent));

    std::vector<cv::cuda::GpuMat> channel_mats;
    channel_mats.push_back(c1);
    channel_mats.push_back(c2);
    channel_mats.push_back(c3);

    cv::cuda::GpuMat result(heights[0], widths[0], CV_8UC3);
    cv::cuda::merge(channel_mats, result);
    if (dst.isGpuMat())
    {
        dst.getGpuMatRef() = result;
    }
    else if (dst.isMat())
    {
        result.download(dst.getMatRef());
    }
    else
    {
        throw std::invalid_argument("unsupport format of cv::OutputArray");
    }
    checkCudaErrors(cudaStreamDestroy(stream));

    return true;
}

bool CudaJpegDecode::Decode(const std::vector<const uchar *> &images, const std::vector<size_t> lengths, cv::OutputArray &dst)
{
    checkCudaErrors(cudaSetDevice(device->get_device_id()));
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<nvjpegImage_t> batch_out;
    std::vector<const uchar *> batch_images;
    std::vector<size_t> batch_img_size;
    std::vector<std::vector<cv::cuda::GpuMat>> batch_channel_mats;
    auto img_iter = images.begin();
    auto img_len_iter = lengths.begin();
    for (int i = 0; i < batch_size_; i++)
    {
        if (img_iter == images.end())
        {
            std::cerr << "Image list is too short to fill the batch, adding files "
                         "from the beginning of the image list"
                      << std::endl;
            img_iter = images.begin();
            img_len_iter = lengths.begin();
        }

        nvjpegImage_t out_temp;
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
        {
            out_temp.channel[c] = nullptr;
            out_temp.pitch[c] = 0;
        }

        int widths[NVJPEG_MAX_COMPONENT];
        int heights[NVJPEG_MAX_COMPONENT];
        int channels;
        nvjpegChromaSubsampling_t subsampling;

        checkCudaErrors(nvjpegGetImageInfo(nvjpeg_handle_, *img_iter, *img_len_iter, &channels, &subsampling, widths, heights));
        int mul = 1;
        if (out_fmt_ == NVJPEG_OUTPUT_RGBI || out_fmt_ == NVJPEG_OUTPUT_BGRI)
        {
            channels = 1;
            mul = 3;
        }
        else if (out_fmt_ == NVJPEG_OUTPUT_RGB || out_fmt_ == NVJPEG_OUTPUT_BGR)
        {
            channels = 3;
            widths[1] = widths[2] = widths[0];
            heights[1] = heights[2] = heights[0];
        }

        cv::cuda::GpuMat c1(heights[0], widths[0], CV_8UC1), c2(heights[0], widths[0], CV_8UC1), c3(heights[0], widths[0], CV_8UC1);
        out_temp.channel[0] = (unsigned char *)c1.cudaPtr();
        out_temp.pitch[0] = c1.step;
        out_temp.channel[1] = (unsigned char *)c2.cudaPtr();
        out_temp.pitch[1] = c2.step;
        out_temp.channel[2] = (unsigned char *)c3.cudaPtr();
        out_temp.pitch[2] = c3.step;

        batch_channel_mats.emplace_back(std::vector<cv::cuda::GpuMat>{c1, c2, c3});
        batch_out.push_back(out_temp);

        batch_images.push_back(*img_iter);
        batch_img_size.push_back(*img_len_iter);
        img_iter++;
        img_len_iter++;
    }

    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
    checkCudaErrors(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    checkCudaErrors(cudaEventRecord(startEvent, stream));
    checkCudaErrors(nvjpegDecodeBatched(nvjpeg_handle_, nvjepg_state_, batch_images.data(), batch_img_size.data(), batch_out.data(), stream));
    checkCudaErrors(cudaEventSynchronize(stopEvent));

    std::vector<cv::cuda::GpuMat> out_results;
    for (auto &channels : batch_channel_mats)
    {
        int width = channels.at(0).cols;
        int height = channels.at(0).rows;
        cv::cuda::GpuMat result(height, width, CV_8UC3);
        cv::cuda::merge(channels, result);
        out_results.push_back(result);
    }

    if (dst.isGpuMatVector())
    {
        dst.getGpuMatVecRef() = out_results;
    }
    else if (dst.isMatVector())
    {
        dst.create(out_results.size(), 1, out_results.at(0).type());
        std::vector<cv::Mat> dst_vec;
        for (int j = 0; j < out_results.size(); j++)
        {
            cv::Mat temp;
            out_results.at(j).download(temp);
            dst.getMatRef(j) = temp;
        }
    }
    else
    {
        throw std::invalid_argument("Only support std::vector<cv::Mat> or std::vector<cv::cuda::GpuMat>");
    }
    checkCudaErrors(cudaStreamDestroy(stream));

    return true;
}

int CudaJpegDecode::host_malloc(void **p, size_t s, unsigned int f)
{
    return (int)cudaHostAlloc(p, s, f);
}

int CudaJpegDecode::host_free(void *p)
{
    return (int)cudaFreeHost(p);
}

int CudaJpegDecode::dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

int CudaJpegDecode::dev_free(void *p)
{
    return (int)cudaFree(p);
}

int CudaJpegDecode::ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int CudaJpegDecode::GpuGetMaxGflopsDeviceId()
{
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() CUDA error:"
                " no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        int computeMode = -1, major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != cudaComputeModeProhibited)
        {
            if (major == 9999 && minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc =
                    ConvertSMVer2Cores(major, minor);
            }
            int multiProcessorCount = 0, clockRate = 0;
            checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
            cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
            if (result != cudaSuccess)
            {
                // If cudaDevAttrClockRate attribute is not supported we
                // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                if (result == cudaErrorInvalidValue)
                {
                    clockRate = 1;
                }
                else
                {
                    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
                            static_cast<unsigned int>(result), cudaGetErrorName(result));
                    exit(EXIT_FAILURE);
                }
            }
            uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf > max_compute_perf)
            {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        }
        else
        {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count)
    {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() CUDA error:"
                " all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}

Jpeg_Decoder_nv::Jpeg_Decoder_nv(int max_wait, int max_decoder) : Jpeg_Decoder(max_wait, max_decoder) {
    format=nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGR;
};

Jpeg_Decoder_nv::~Jpeg_Decoder_nv() {

};
void Jpeg_Decoder_nv::init(std::vector<std::shared_ptr<Device_Handle>> devices_handles, int init_cnt)
{

    if(this->devices_handles.size()>0)
        return;
    if(devices_handles.size()<=0)
        return;
    std::unique_lock lock(mutex);
    //std::cout<<"init jped decoder pool start"<<std::endl;
    this->format=format;
    this->devices_handles=devices_handles;
    this->decoders.resize(max_decoder);
    this->decoder_states.resize(max_decoder);
    for(int i=0;i<init_cnt;i++){        
        std::shared_ptr<Device_Handle> device=devices_handles[i%devices_handles.size()];
   //     std::cout<<"init jped decoder "<<i<<"\t"<<device_id<<std::endl;
        this->decoders[i].DeviceInit(format,device);
        
        this->decoder_states[i]=0;
    }
    this->current_cnt=init_cnt;
 };
std::shared_ptr<QyImage> Jpeg_Decoder_nv::decode(const std::vector<unsigned char> &data)
{
    std::shared_ptr<QyImage> result;
    if (current_cnt <= 0)
        return result;
    long start_time = get_time();
    int select_idx = -1;

    while (true)
    {

        select_idx = get_instance_idx();
        if (select_idx >= 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        long current_time = get_time();
        if (current_time - start_time > this->max_wait)
        {
            if (current_cnt < this->decoders.size())
            {
                select_idx = add_instance(nullptr);
                break;
            }
        }
    }
    if (select_idx >= 0)
    {
        cv::cuda::GpuMat dst;
        bool res = decoders[select_idx].Decode(data.data(), data.size(), dst);
        result = from_data(dst.data, dst.elemSize(), dst.step, dst.cols, dst.rows, decoders[select_idx].device, false);
        //        std::cout<<"decoce jpeg image by "<<select_idx<<"\t"<<res<<" width:"<<dst.cols()<<" height:"<<dst.rows()<<std::endl;
        std::unique_lock lock(mutex);
        decoder_states[select_idx] = 0;
        if (dst.cols <= 0 || dst.rows <= 0)
        {
            return result;
        }
        return result;
    }
    return result;
};
int Jpeg_Decoder_nv::add_instance(std::shared_ptr<Device_Handle> device)
{
    std::unique_lock lock(mutex);
    if (device == nullptr)
        device = devices_handles[current_cnt % devices_handles.size()];
    //  std::cout<<" add instance "<<device_id<<"\t"<<current_cnt<<std::endl;
    decoders[current_cnt].DeviceInit(format, device);
    decoder_states[current_cnt] = 1;
    current_cnt += 1;
    return current_cnt - 1;
};
int Jpeg_Decoder_nv::get_instance_idx()
{
    std::unique_lock lock(mutex);
    for (int i = 0; i < current_cnt; i++)
    {
        if (decoder_states[i] == 0)
        {
            decoder_states[i] = 1;
            return i;
        }
    }
    return -1;
};

extern "C" std::shared_ptr<Jpeg_Decoder> get_jpeg_decoder(int max_wait,int max_decoder){
    std::shared_ptr<Jpeg_Decoder_nv> decoder=std::make_shared<Jpeg_Decoder_nv>(max_wait,max_decoder);
    std::shared_ptr<Jpeg_Decoder> res=decoder;
    return res;
};
