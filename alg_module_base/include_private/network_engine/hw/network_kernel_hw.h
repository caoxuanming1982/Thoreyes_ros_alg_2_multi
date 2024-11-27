#ifndef __NETWORK_KERNEL_BM_H__
#define __NETWORK_KERNEL_BM_H__
#define USE_OPENCV

#include <iostream>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

#include <thread>
#include <mutex>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <shared_mutex>
#include "common.h"
#include "network_engine/device_handle.h"
#include "network_engine/network_kernel.h"
#include "cv_lib/type_def.h"
#include "cv_lib/hw/type_def_hw.h"

class DevicePtr
{
public:
    size_t size = 0;
    void *data = nullptr;

    DevicePtr() ;
    ~DevicePtr();

    void resize(int size_in);

    void from_model_input(aclmdlDesc *modelDesc_, int idx);
    void from_model_output(aclmdlDesc *modelDesc_, int idx);
    void free();

    aclDataBuffer *get_buffer();
};

class Network
{
    size_t modelMemSize_ = 0;
    size_t modelWeightSize_ = 0;
    char *omModelPath = nullptr;
    void *modelMemPtr_ = nullptr;
    void *modelWeightPtr_ = nullptr;
    uint32_t modelId_ = 4294967295;
    std::shared_ptr<Device_Handle> current_handle;

    aclmdlDesc *modelDesc_ = nullptr;

    std::vector<DevicePtr> input_buffer;
    std::vector<DevicePtr> output_buffer;
    aclmdlDataset *inputs = nullptr;
    aclmdlDataset *outputs = nullptr;

    bool use_fp16 = false;

public:
    Network(std::string path, std::shared_ptr<Device_Handle> handle);

    ~Network();
    
    bool init();

    std::vector<Shape_t> get_input_shapes();

    bool prepare_input_buffer();

    bool free_input_buffer();

    bool prepare_output_buffer();

    bool free_output_buffer();

    bool forward(std::vector<std::shared_ptr<QyImage_hw>> &inputs_, std::vector<Output> &outputs_, std::vector<cv::Scalar> &scale, std::vector<cv::Scalar> &offset);

    Network *copy(std::shared_ptr<Device_Handle> handle);
};

class Network_kernel_hw : public Network_kernel
{

    std::vector<Network *> net_instances;

public:
    Network_kernel_hw(std::shared_ptr<Device_Handle> handle, std::string file_path, std::string model_name, std::vector<Shape_t> &input_shapes, int max_instance = 2);
    virtual ~Network_kernel_hw()
    {
        for (int i = 0; i < this->net_instances.size(); i++)
        {
            delete this->net_instances[i];
            this->net_instances[i] = nullptr;
        }
        this->net_instances.clear();
    };
    virtual int forward(std::vector<std::shared_ptr<QyImage>> &inputs, std::vector<Output> &outputs);
    virtual std::vector<Shape_t> get_input_shapes();
};

#endif