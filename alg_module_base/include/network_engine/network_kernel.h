#ifndef __NETWORK_KERNEL_H__
#define __NETWORK_KERNEL_H__
#define USE_OPENCV

#include<iostream>

#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>

#include<thread>
#include<mutex>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <shared_mutex>
#include "common.h"
#include "network_engine/device_handle.h"
#include "cv_lib/type_def.h"
#include <filesystem>

class Network_kernel{
public:
    std::shared_ptr<Device_Handle> handle_;
    std::string file_path;
    std::string model_name;
    std::vector<Shape_t> cache_inputs_shapes;
    std::vector<std::shared_mutex*> instance_mutex;
    int cnt=0;
    int max_instance=1;
    std::vector<cv::Scalar> input_scale;
    std::vector<cv::Scalar> input_offset;

    Network_kernel(std::shared_ptr<Device_Handle> handle,std::string file_path,std::string model_name,std::vector<Shape_t>& input_shapes,int max_instance=2);
    virtual ~Network_kernel();
    virtual int forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs)=0;
    virtual std::vector<Shape_t> get_input_shapes();
    virtual int check_need_scale_offset(int idx);
};
extern "C" Network_kernel* get_network_kernel(std::shared_ptr<Device_Handle> handle,std::string file_path,std::string model_name,std::vector<Shape_t>& input_shapes,int max_instance=2);
extern "C" void free_network_kernel(Network_kernel* instance);

extern "C" std::string convert_model_path(std::string model_path);
extern "C" bool global_init();
#endif