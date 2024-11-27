#ifndef __ALG_MODULE_BASE_H__
#define __ALG_MODULE_BASE_H__
#include<iostream>

#include<vector>
#include<map>

#include "cv_lib/type_def.h"
#include "network_engine/device_handle.h"

#include<thread>
#include<mutex>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <shared_mutex>
#include "tr_cfg_type_base.h"
#include "network_engine/network_parallel.h"
#include "network_engine/device_handle.h"
#include "cv_lib/type_def.h"
#include "inout_type.h"

#ifdef _WIN32
    #define CLASS_EXPORT _declspec(dllexport)
    #define FUNC_EXPORT
    #define GLOBAL_FUNC_EXPORT _declspec(dllexport)
#elif defined(__linux__)
    #define CLASS_EXPORT
    #define FUNC_EXPORT __attribute__((visibility("default")))
    #define GLOBAL_FUNC_EXPORT __attribute__((visibility("default")))
#endif

class CLASS_EXPORT Alg_Module_Base{
    static std::map<std::string,std::shared_ptr<Alg_Module_Base>> function_module;
public:
    FUNC_EXPORT static void set_function_module(std::string name,std::shared_ptr<Alg_Module_Base> module);
    FUNC_EXPORT static std::shared_ptr<Alg_Module_Base> get_function_module(std::string name);
    FUNC_EXPORT static std::vector<std::string> get_function_module_name();

    Alg_Module_Base();
    FUNC_EXPORT virtual ~Alg_Module_Base();
    FUNC_EXPORT virtual std::string get_module_name()=0;                                        //获取模块名
    FUNC_EXPORT virtual int get_module_tick_interval()=0;                                       //获取模块运行的间隔
    FUNC_EXPORT virtual bool get_input_description(std::shared_ptr<InputOutput_cfg>& res)=0;    //获取模块输入输出的描述

    FUNC_EXPORT virtual bool init_from_root_dir(std::string root_dir)=0;                        //从指定路径初始化模块

    FUNC_EXPORT virtual std::map<std::string,float> check_model_util()=0;                       //获取模块中每个模型的负载
    FUNC_EXPORT virtual bool increase_model_instane(std::string model_name,int device_id)=0;    //增加模块中指定模型的实例数
    FUNC_EXPORT virtual bool reduce_model_instane(std::string model_name)=0;                    //减少模块中指定模型的实例数

    FUNC_EXPORT virtual void set_device_ids(std::vector<int> device_ids)=0;                     //设置可使用的计算卡核心的id，只能调用一次
    FUNC_EXPORT virtual void set_device_handles(std::vector<std::shared_ptr<Device_Handle>> device_ids)=0;         //设置可使用的计算卡核心的handle，只能调用一次

    FUNC_EXPORT virtual bool load_module_cfg(std::string cfg_path)=0;                           //加载模块配置文件
    FUNC_EXPORT virtual bool load_model_cfg(std::string cfg_path,std::string model_name)=0;     //加载模型配置文件
    FUNC_EXPORT virtual bool load_channel_cfg(std::string channel_name,std::string cfg_path)=0; //加载通道配置文件
    FUNC_EXPORT virtual bool init_channal_data(std::string channel_name)=0;                     //初始化通道独立的记忆数据
    FUNC_EXPORT virtual bool load_model(std::string model_path,std::string model_name,std::string model_cfg="")=0;  //加载模型

    FUNC_EXPORT virtual std::shared_ptr<Module_cfg_base> get_module_cfg()=0;                    //获取模块配置文件
    FUNC_EXPORT virtual std::shared_ptr<Model_cfg_base> get_model_cfg(std::string model_name)=0;    //获取模型配置文件
    FUNC_EXPORT virtual std::shared_ptr<Channel_cfg_base> get_channel_cfg(std::string channel_name)=0;  //获取通道配置文件
    FUNC_EXPORT virtual std::shared_ptr<Channel_data_base> get_channal_data(std::string channel_name)=0;    //获取通道独立的记忆数据

    FUNC_EXPORT virtual bool reset_channal_data(std::string channel_name)=0;                    //重置指定通道的记忆数据

    FUNC_EXPORT virtual bool forward(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& output)=0;      //模块推理
    FUNC_EXPORT virtual bool filter(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& output)=0;       //结果过滤
    FUNC_EXPORT virtual bool display(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& filter_output)=0;      //可视化


};


#endif