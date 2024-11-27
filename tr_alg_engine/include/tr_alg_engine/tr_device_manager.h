#ifndef __TR_DEVICE_MANAGER_H__
#define __TR_DEVICE_MANAGER_H__


#include<vector>
#include <rclcpp/rclcpp.hpp>
#include<chrono>
#include <shared_mutex>

#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#include<network_engine/device_handle.h>
#include "tr_alg_interfaces/srv/get_avaliable_device.hpp"
#include "tr_alg_interfaces/msg/device_usage.hpp"
#include "common.h"
#include "tr_alg_node_base.h"
#include "moniter/moniter.h"
using namespace std;  



class Device_info{
public:
    int device_id=-1;
    int total_mem_mbytes=0;
    int avaliable_mem_mbytes=0;
    int util=0;

    float protect_mem=500;

    bool check_memory(float mem_reques);

    bool check_util(float util_request);

    std::string str();

};

class Host_info{
public:
        

    float total_mem_mbytes=0;
    float avaliable_mem_mbytes=0;
    float cpu_max=0;
    float cpu_current_average=0;

    float protect_mem=500;

    bool check_memory(float mem_reques);

    bool check_util(float util_request);
    std::string str();


};

class Task{
public:
    int count_down=0;
    int max_count=10;
    int memory_mbytes=0;
    int tpu_memory_mbytes=0;
    int tpu_load=0;
    int cpu_load=0;
    int device_id=-1;
    Task();
    Task(int memory_mbytes,int tpu_memory_mbytes,int cpu_load,int tpu_load);

    Task(const Task & other);
    void tick();
    bool timeout();
    void operator=(const Task & other);

};

class Task_prepare{
public:
    std::vector<Task> tasks;
    void push_task(Task& task);
    void tick();
    std::map<int,float> check_new_task_with_util_res(Task& new_task,std::vector<Device_info> devices_info,Host_info host_info,std::vector<unsigned int> device_idx,bool force_get=false);
    int check_new_task(Task& new_task,std::vector<Device_info> devices_info,Host_info host_info,std::vector<unsigned int> device_idx,bool force_get=false);

    void add_prepare_source(std::vector<Device_info>& devices_info,Host_info& host_info);

};

class Alg_Node_Device_Manager:public Alg_Node_Base{
    std::vector<Device_info> devices_info;
    Host_info host_info;

    std::vector<std::shared_ptr<Device_Handle>> devices_handle;

    std::vector<unsigned int > devices_card_idx;
    std::map<unsigned int,std::vector<unsigned int>> device_card_map;

    

    unsigned int card_num=0;

    bool is_first=true;
    rclcpp::TimerBase::SharedPtr timer_;

    Task_prepare tasks;
    std::shared_mutex mutex;

    rclcpp::Publisher<tr_alg_interfaces::msg::DeviceUsage>::SharedPtr device_usage_publisher;
    rclcpp::Service<tr_alg_interfaces::srv::GetAvaliableDevice>::SharedPtr get_device_Server;

    std::shared_ptr<Moniter> moniter;

public:

    Alg_Node_Device_Manager(std::string dev_platform_name);
    void init_devices_handle();
    void reset_devices_handle();
    void init_devices_info();
    void init_host_info();

    void update_devices_info();
    void update_host_info();
    void update_tasks();
    void update();

    void get_device(const tr_alg_interfaces::srv::GetAvaliableDevice::Request::SharedPtr request,
        const tr_alg_interfaces::srv::GetAvaliableDevice::Response::SharedPtr response);
};


extern "C" std::shared_ptr<Alg_Node_Device_Manager> get_device_manager_node();



#endif
