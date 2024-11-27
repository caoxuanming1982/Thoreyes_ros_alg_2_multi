#ifndef __TR_ALG_ENGINE_H__
#define __TR_ALG_ENGINE_H__
#include<vector>
#include <rclcpp/rclcpp.hpp>
#include<chrono>
#include <shared_mutex>
#include<fstream>
#include<sstream>
#include<future>
#include<list>

#include "alg_engine.h"
#include "tr_alg_node_base.h"
#include "jpeg_decode/jpeg_decoder.h"
#include "tr_interfaces/msg/tr_image_algo_result.hpp"
#include "tr_alg_interfaces/msg/results.hpp"
#include "tr_alg_interfaces/msg/channel_list.hpp"
#include "tr_alg_interfaces/msg/device_usage.hpp"
#include "tr_alg_interfaces/srv/get_avaliable_device.hpp"
#include "tr_alg_interfaces/srv/set_channel_config.hpp"
#include "tr_alg_interfaces/srv/set_module_enable.hpp"
#include "tr_alg_interfaces/srv/get_module_enable.hpp"
#include "tr_alg_interfaces/srv/get_channel_status.hpp"

#include "error_type.h"
#include <network_engine/device_handle.h>

using std::placeholders::_1;
using std::placeholders::_2;

class Engine_time_counter{
    std::vector<long long> inference_used_ms;
    std::vector<long long> decode_used_ms;
    std::vector<long long> publish_used_ms;
    int max_save=100;
    
    std::map<std::string,long long> last_inference_time;
    std::map<std::string,long long> last_decode_time;
    std::map<std::string,long long> last_publish_time;
    std::shared_mutex mutex;
public:
    std::string module_name="engine";
    bool enable=true;
    Engine_time_counter(){
    }

    void start_inference(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        last_inference_time[channel_name]=timestamp;
    };
    void end_inference(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        std::unique_lock lock(mutex);
        inference_used_ms.push_back(timestamp-last_inference_time[channel_name]);
        if(inference_used_ms.size()>max_save){
            inference_used_ms.erase(inference_used_ms.begin());
        }
    };
    void start_decode(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        last_decode_time[channel_name]=timestamp;
    };
    void end_decode(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        std::unique_lock lock(mutex);
        decode_used_ms.push_back(timestamp-last_decode_time[channel_name]);
        if(decode_used_ms.size()>max_save){
            decode_used_ms.erase(decode_used_ms.begin());
        }
    };

    void start_publish(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        last_publish_time[channel_name]=timestamp;
    };
    void end_publish(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        std::unique_lock lock(mutex);
        publish_used_ms.push_back(timestamp-last_publish_time[channel_name]);
        if(publish_used_ms.size()>max_save){
            publish_used_ms.erase(publish_used_ms.begin());
        }
    };

    std::string to_string(){
        std::string res;
        if(enable==false){
            return res;
        }
        res+="&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";
        res+=module_name+"\n";
        int sum_ms=0;
        for(int i=0;i<inference_used_ms.size();i++){
            sum_ms+=inference_used_ms[i];
        }
        if(inference_used_ms.size()>1)
            sum_ms/=inference_used_ms.size();
        int sum_ms_high=0;
        int n_sum_ms_high=0;
        for(int i=0;i<inference_used_ms.size();i++){
            if(inference_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=inference_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="inference:  mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";

        sum_ms=0;
        for(int i=0;i<decode_used_ms.size();i++){
            sum_ms+=decode_used_ms[i];
        }
        if(decode_used_ms.size()>1)
            sum_ms/=decode_used_ms.size();
        sum_ms_high=0;
        n_sum_ms_high=0;
        for(int i=0;i<decode_used_ms.size();i++){
            if(decode_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=decode_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="decode:     mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";
        sum_ms=0;
        for(int i=0;i<publish_used_ms.size();i++){
            sum_ms+=publish_used_ms[i];
        }
        if(publish_used_ms.size()>1)
            sum_ms/=publish_used_ms.size();
        sum_ms_high=0;
        n_sum_ms_high=0;
        for(int i=0;i<publish_used_ms.size();i++){
            if(publish_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=publish_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="publish:    mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";

        return res;
    };


};


class Counter{
public:
    std::list<long long> timestamps;
    int max_size=10;
    std::shared_mutex mutex;
    void tick(long long timestamp){
        this->timestamps.push_back(timestamp);
        if(this->timestamps.size()>max_size){
            this->timestamps.pop_front();
        }
    };
    float get_fps(){
        if(timestamps.size()<5)
            return 0;
        else{
            return 1000.0/((timestamps.back()-timestamps.front())/(timestamps.size()-1));
        }
    };
    std::string get_intervals(){
        std::string res;
        if(timestamps.size()>1){
            std::vector<int> timestamps_t(timestamps.begin(),timestamps.end());
            for(int i=0;i<timestamps_t.size()-1;i++){
                res+="\t"+Num2string<int>(timestamps_t[i+1]-timestamps_t[i]);
            }
        }
        return res;
    };
};


class Tr_Alg_Engine_module : public Alg_Node_Base{
protected:
    Alg_Engine engine;

    std::map<int,std::shared_ptr<Device_Handle>> device_handles_all;
    std::vector<std::shared_ptr<Device_Handle>> device_handles_inuse;
    std::shared_ptr<Jpeg_Decoder> jpeg_decoder;

    Engine_time_counter engine_time_counter; 

    rclcpp::Publisher<tr_interfaces::msg::TrImageAlgoResult>::SharedPtr publisher_primary;
    std::map<std::string,rclcpp::Publisher<tr_alg_interfaces::msg::Results>::SharedPtr> raw_publisher;

    std::map<std::string,rclcpp::Subscription<tr_interfaces::msg::RvFrame>::SharedPtr> Subscription_;
    std::map<std::string,std::shared_ptr<Counter>> counters;
    std::map<std::string,std::shared_ptr<Counter>> counters_real_time;

    rclcpp::Subscription<tr_alg_interfaces::msg::ChannelList>::SharedPtr channel_list_subscription;
    rclcpp::Subscription<tr_alg_interfaces::msg::DeviceUsage>::SharedPtr device_usage_subscription;

    rclcpp::Service<tr_alg_interfaces::srv::SetChannelConfig>::SharedPtr set_channel_config_service;
    rclcpp::Service<tr_alg_interfaces::srv::SetModuleEnable>::SharedPtr set_module_enable_service;
    rclcpp::Service<tr_alg_interfaces::srv::GetModuleEnable>::SharedPtr get_module_enable_service;
    rclcpp::Service<tr_alg_interfaces::srv::GetChannelStatus>::SharedPtr get_channel_status_service;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::CallbackGroup::SharedPtr callback_group;
    rclcpp::CallbackGroup::SharedPtr recv_image_callback_group;

    bool load_finish=false;

    bool test_modules_mode=false;

    int tick_cnt=0;
    int log_util_interval=10;



public:
    bool show_util=false;

    Tr_Alg_Engine_module(std::string dev_platform_name);
 
    bool init(std::string submodule_dir="",std::string requirement_dir="");
    virtual bool init_(std::string submodule_dir,std::string requirement_dir);

    void update_device_handles(const tr_alg_interfaces::msg::DeviceUsage::ConstPtr& msg);

    void tick();

    void updata_channel(const tr_alg_interfaces::msg::ChannelList::ConstPtr & channel_list);

    virtual ~Tr_Alg_Engine_module();

    void set_channel_cfg(std::string channel_name,std::string cfg_path);
    bool load_module_from_dir(std::string lib_dir,std::string requirement_dir);
    bool load_module_from_path(std::string lib_path,std::string requirement_dir);
    void set_device_handles(std::vector<int>& device_ids);

    bool check_loaded_modules();

    std::shared_ptr<Device_Handle> get_random_device_handle();
    std::shared_ptr<QyImage> trans_from_rv_frame(const tr_interfaces::msg::RvFrame::ConstPtr &frame_data);  
    

    bool trans_from_rv_frame(const tr_interfaces::msg::RvFrame::ConstPtr & frame_data,std::map<std::string,std::shared_ptr<InputOutput>>& input_data);
    bool trans_from_rv_frame_cloud(const tr_interfaces::msg::RvFrame::ConstPtr & frame_data,std::map<std::string,std::shared_ptr<InputOutput>>& input_data);
    
    void recv_image_callback(const tr_interfaces::msg::RvFrame::ConstPtr & frame_data,std::string channel_name);

    bool trans_to_raw_publish_struct(std::string channel_name,long long timestamp,std::shared_ptr<Publish_data> output,tr_alg_interfaces::msg::Results & raw_result);

    bool trans_to_event_publish_struct(const tr_interfaces::msg::RvFrame::ConstPtr & frame_data,std::shared_ptr<Publish_data> output,std::vector<tr_interfaces::msg::TrImageAlgoResult>& result);
    bool debug_show_event_publish_data(std::string channel_name,std::vector<tr_interfaces::msg::TrImageAlgoResult>& result,long long timestamp_in);

    void set_channel_config_callback(const tr_alg_interfaces::srv::SetChannelConfig::Request::SharedPtr request,
        const tr_alg_interfaces::srv::SetChannelConfig::Response::SharedPtr response);
    void set_module_enable_callback(const tr_alg_interfaces::srv::SetModuleEnable::Request::SharedPtr request,
        const tr_alg_interfaces::srv::SetModuleEnable::Response::SharedPtr response);
    void get_module_enable_callback(const tr_alg_interfaces::srv::GetModuleEnable::Request::SharedPtr request,
        const tr_alg_interfaces::srv::GetModuleEnable::Response::SharedPtr response);

    void get_channel_status(const tr_alg_interfaces::srv::GetChannelStatus::Request::SharedPtr request,
        const tr_alg_interfaces::srv::GetChannelStatus::Response::SharedPtr response);
};


extern "C" std::shared_ptr<Tr_Alg_Engine_module> get_engine_node();
#endif