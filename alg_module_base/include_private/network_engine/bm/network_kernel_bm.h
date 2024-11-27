#ifndef __NETWORK_KERNEL_BM_H__
#define __NETWORK_KERNEL_BM_H__
#define USE_OPENCV

#include<iostream>

#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>


#include<bmruntime.h>
#include<bmruntime_cpp.h>
#include<bmcv_api.h>
using namespace bmruntime;

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



class Network_kernel_bm:public Network_kernel{

	Context* p_ctx = nullptr;
	std::vector<Network*> net_instances;

public:

    Network_kernel_bm(std::shared_ptr<Device_Handle> handle,std::string file_path,std::string model_name,std::vector<Shape_t>& input_shapes,int max_instance=2);
    virtual ~Network_kernel_bm(){
        for (int i=0;i<this->net_instances.size();i++){
            delete this->net_instances[i];
            this->net_instances[i]=nullptr;
        }
        this->net_instances.clear();
        if(p_ctx!=nullptr){
            delete p_ctx;
            p_ctx=nullptr;
        }
    };
    virtual int forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs);
    virtual std::vector<Shape_t> get_input_shapes();

};


#endif