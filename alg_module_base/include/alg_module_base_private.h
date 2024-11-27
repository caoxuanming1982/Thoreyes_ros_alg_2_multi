#ifndef __ALG_MODULE_BASE_PRIVATE_H__
#define __ALG_MODULE_BASE_PRIVATE_H__
#include<iostream>

#include<vector>
#include<map>
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>

#include<thread>
#include<mutex>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <shared_mutex>
#include "tr_cfg_type_base.h"
#include "network_engine/network_parallel.h"
#include "alg_module_base.h"

#ifdef HAVE_OMP
#include <omp.h>
#endif


class Alg_Module_Base_private:public Alg_Module_Base{
protected:

    int tick_interval_ms=1000;
    std::string node_name;
    int tick_cnt=0;

    std::vector<int> device_ids;  
    std::map<int,std::shared_ptr<Device_Handle>> device_handle_cache;

    std::map<std::string,std::shared_ptr<Network_parallel>> networks;
    std::map<std::string,std::shared_ptr<Model_cfg_base>> model_cfg;
    std::map<std::string,std::shared_ptr<Channel_data_base>> channel_data;
    std::map<std::string,std::shared_ptr<Channel_cfg_base>> channel_cfg;
    std::shared_ptr<Module_cfg_base> module_cfg;

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);            //加载模块配置文件，如果继承Module_cfg_base子类进行实现，则此函数也需要覆盖
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);            //加载模型配置文件，如果继承Model_cfg_base子类进行实现，则此函数也需要覆盖
    virtual std::shared_ptr<Channel_cfg_base> load_channel_cfg_(std::string channel_name,std::string cfg_path);            //加载通道配置文件，如果继承Channel_cfg_base子类进行实现，则此函数也需要覆盖
    virtual std::shared_ptr<Channel_data_base> init_channal_data_(std::string channel_name);    //初始化通道独立的需要记忆的变量，如果继承Channel_data_base子类进行实现，则此函数也需要覆盖

    bool load_model_(std::string model_path,std::string model_name,std::string model_cfg="");

    void set_module_cfg(std::shared_ptr<Module_cfg_base> cfg);
    void set_model_cfg(std::string model_name,std::shared_ptr<Model_cfg_base> model_cfg);
    void set_channel_cfg(std::shared_ptr<Channel_cfg_base> cfg);
    void set_channal_data(std::string channel_name,std::shared_ptr<Channel_data_base> data);

    std::shared_ptr<Network_parallel> get_model_instance(std::string model_name);

    bool get_device( std::shared_ptr<Device_Handle>&handle); 



    std::shared_mutex mutex;
    std::shared_mutex mutex_channel_cfg;
    std::shared_mutex mutex_channel_data;


public:
    Alg_Module_Base_private(std::string node_name);
    virtual ~Alg_Module_Base_private();

    void set_device_ids(std::vector<int> device_ids);

    void set_device_handles(std::vector<std::shared_ptr<Device_Handle>> device_ids);

    std::map<std::string,float> check_model_util();
    bool increase_model_instane(std::string model_name,int device_id);
    bool reduce_model_instane(std::string model_name);


    bool load_module_cfg(std::string cfg_path);
    bool load_model_cfg(std::string cfg_path,std::string model_name);
    bool load_channel_cfg(std::string channel_name,std::string cfg_path);
    bool init_channal_data(std::string channel_name);
    bool load_model(std::string model_path,std::string model_name,std::string model_cfg="");

    std::shared_ptr<Module_cfg_base> get_module_cfg();
    std::shared_ptr<Model_cfg_base> get_model_cfg(std::string model_name);
    std::shared_ptr<Channel_cfg_base> get_channel_cfg(std::string channel_name);                    //获取通道配置文件
    std::shared_ptr<Channel_data_base> get_channal_data(std::string channel_name);                  //获取通道记忆的数据


    std::string get_module_name();
    int get_module_tick_interval();

    bool reset_channal_data(std::string channel_name);
    bool get_input_description(std::shared_ptr<InputOutput_cfg>& res);

    virtual bool init_from_root_dir(std::string  root_dir)=0;    //从指定路径初始化模块，需要实现
    virtual bool forward(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& output)=0;  //模型推理，需要实现
    virtual bool filter(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& output)=0;   //结果过滤，需要实现
    virtual bool display(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& filter_output)=0;  //可视化，主要是画框图和扣车牌，需要实现

};

#endif