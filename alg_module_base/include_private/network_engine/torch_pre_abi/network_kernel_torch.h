#ifndef __NETWORK_KERNEL_BM_H__
#define __NETWORK_KERNEL_TORCH_H__
#define USE_OPENCV

#include<iostream>

#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>

#include <torch/torch.h>
#include <torch/jit.h>


#include<thread>
#include<mutex>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <shared_mutex>
#include "common.h"
#include "network_engine/device_handle.h"
#include "network_engine/network_kernel.h"
#include "cv_lib/type_def.h"


class Network_kernel_torch:public Network_kernel{

    std::vector<torch::jit::script::Module> net_instances;

public:

    Network_kernel_torch(std::shared_ptr<Device_Handle> handle,std::string file_path,std::string model_name,std::vector<Shape_t>& input_shapes,int max_instance=2);
    virtual ~Network_kernel_torch(){
        this->net_instances.clear();
    };
    virtual int forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs);
    virtual std::vector<Shape_t> get_input_shapes();

};


#endif