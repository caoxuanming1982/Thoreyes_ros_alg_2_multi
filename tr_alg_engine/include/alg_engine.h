#ifndef __ALG_ENGINE_H__
#define __ALG_ENGINE_H__

#include <dlfcn.h>  
#include <iostream>
#include "alg_module_base.h"
#include "tr_cfg_type_base.h"

#include "inout_type.h"
#include "publish_cfg_base.h"
#include "post_process_cfg_base.h"

#include <network_engine/device_handle.h>

#include<fstream>
#include<sstream>

#include <map>
#include<unordered_map>
#include <vector>
#include <thread>
#include <future>
#include <string>

#include <shared_mutex>
#include <thread>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#define __WITH_TRY_CATCH__

typedef Alg_Module_Base* (*Create_func)();
typedef void (*Destory_func)(Alg_Module_Base*);

class Module_time_counter{
    std::vector<long long> inference_used_ms;
    std::vector<long long> forward_used_ms;
    std::vector<long long> display_used_ms;
    int max_save=100;
    
    std::map<std::string,long long> last_inference_time;
    std::map<std::string,long long> last_forward_time;
    std::map<std::string,long long> last_display_time;
    std::shared_mutex mutex;
public:
    std::string module_name;
    bool enable=true;
    Module_time_counter(){
    }

    Module_time_counter(std::string module_name){
        this->module_name=module_name;
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
    void start_filter(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        last_forward_time[channel_name]=timestamp;
    };
    void end_filter(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        std::unique_lock lock(mutex);
        forward_used_ms.push_back(timestamp-last_forward_time[channel_name]);
        if(forward_used_ms.size()>max_save){
            forward_used_ms.erase(forward_used_ms.begin());
        }
    };

    void start_display(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        last_display_time[channel_name]=timestamp;
    };
    void end_display(std::string channel_name,long long timestamp){
        if(enable==false){
            return;
        }
        std::unique_lock lock(mutex);
        display_used_ms.push_back(timestamp-last_display_time[channel_name]);
        if(display_used_ms.size()>max_save){
            display_used_ms.erase(display_used_ms.begin());
        }
    };

    std::string to_string(){
        std::string res;
        if(enable==false){
            return res;
        }
        res+="$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
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
        for(int i=0;i<forward_used_ms.size();i++){
            sum_ms+=forward_used_ms[i];
        }
        if(forward_used_ms.size()>1)
            sum_ms/=forward_used_ms.size();
        sum_ms_high=0;
        n_sum_ms_high=0;
        for(int i=0;i<forward_used_ms.size();i++){
            if(forward_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=forward_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="forward:    mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";
        sum_ms=0;
        for(int i=0;i<display_used_ms.size();i++){
            sum_ms+=display_used_ms[i];
        }
        if(display_used_ms.size()>1)
            sum_ms/=display_used_ms.size();
        sum_ms_high=0;
        n_sum_ms_high=0;
        for(int i=0;i<display_used_ms.size();i++){
            if(display_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=display_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="display:    mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";

        return res;
    };

    std::string to_string_detail(){
        std::string res;
        if(enable==false){
            return res;
        }
        res+="$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
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
        for(int i=0;i<inference_used_ms.size();i++){
            if(inference_used_ms[i]>1.5*sum_ms){
                res+=std::to_string(inference_used_ms[i])+"\t";
            }
        }
        res+="\n";
        

        sum_ms=0;
        for(int i=0;i<forward_used_ms.size();i++){
            sum_ms+=forward_used_ms[i];
        }
        if(forward_used_ms.size()>1)
            sum_ms/=forward_used_ms.size();
        sum_ms_high=0;
        n_sum_ms_high=0;
        for(int i=0;i<forward_used_ms.size();i++){
            if(forward_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=forward_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="forward:    mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";
        for(int i=0;i<forward_used_ms.size();i++){
            if(forward_used_ms[i]>1.5*sum_ms){
                res+=std::to_string(forward_used_ms[i])+"\t";
            }
        }
        res+="\n";

        sum_ms=0;
        for(int i=0;i<display_used_ms.size();i++){
            sum_ms+=display_used_ms[i];
        }
        if(display_used_ms.size()>1)
            sum_ms/=display_used_ms.size();
        sum_ms_high=0;
        n_sum_ms_high=0;
        for(int i=0;i<display_used_ms.size();i++){
            if(display_used_ms[i]>1.5*sum_ms){
                sum_ms_high+=display_used_ms[i];
                n_sum_ms_high+=1;
            }
        }
        if(n_sum_ms_high>1){
            sum_ms_high/=n_sum_ms_high;
        }
        res+="display:  mean:"+std::to_string(sum_ms)+"ms\t n_high:"+std::to_string(n_sum_ms_high)+"\t high mean:"+std::to_string(sum_ms_high)+"ms \n";
        for(int i=0;i<display_used_ms.size();i++){
            if(display_used_ms[i]>1.5*sum_ms){
                res+=std::to_string(display_used_ms[i])+"\t";
            }
        }
        res+="\n";

        return res;

    };

};


class Alg_Engine;
class Inference_graph;
class Alg_Node{
    void *handle=nullptr;
    Create_func create=nullptr;
    Destory_func destory=nullptr;
    std::shared_ptr<Alg_Module_Base> module=nullptr;

    std::shared_ptr<InputOutput_cfg> inout_cfg;
    std::shared_ptr<Publish_cfg> publish_cfg;
    std::shared_ptr<Post_process_cfg_base> post_process_cfg;
    
    std::string lib_path="";
    std::string requirement_dir="";

    friend class Alg_Engine;
    friend class Inference_graph;
public:
    Module_time_counter time_counter;

    std::map<std::string,bool> channel_enable;
    bool enable=false;

    std::string module_name;
    int tick_interval=0;
    int filter_interval=0;
    bool is_real_time=false;
    bool is_func_module=false;

    bool need_cache_for_unreal_time=false;
    bool no_filter=false;

    Alg_Node();
    virtual ~Alg_Node();
    bool init(std::string requirement_root_dir_path);
    bool load(std::string lib_path);

    
};
class Inference_graph;
class Inference_graph_node{
    std::shared_ptr<Alg_Node> node;
    std::set<int> required_node_idx;  
    std::set<int> next_node_idx;  
    bool is_infered=false;
    int idx=0;

    friend class  Inference_graph;
public:
    Inference_graph_node(std::shared_ptr<Alg_Node> node);

};

class Inference_graph{
    std::vector<std::shared_ptr<Inference_graph_node>> nodes;
    std::map<std::string,int> node2idx;


public:
    Inference_graph();
    void insert_node(std::shared_ptr<Alg_Node> node);

    void make_edge();

    std::vector<std::shared_ptr<Alg_Node>> gen_inference_queue();


};

struct Node_require_check_result{

    std::string module_name;
    std::string param_name;
    std::string error_msg;
    
};

struct Publish_data{
    std::string topic_base;
    std::shared_ptr<InputOutput> data;
    int raw_publish=0;
    std::string module_name;
};

struct Request_Model_instance_data{
    std::shared_ptr<Model_cfg_base> cfg;
    std::string module_name;
    std::string model_name;
    int result_device_id=-1;
    bool has_ins=false;
};


class Tick_manager{
    std::map<std::string,long long> last_forward_timestamp;
    std::map<std::string,long long> last_filter_timestamp;
    std::string channel_name;

public:
    Tick_manager(std::string channel_name);
    virtual ~Tick_manager();

    std::vector<std::string> check_need_forward(std::vector<std::shared_ptr<Alg_Node>> nodes,long long timestamp,bool is_real_time=false);

    std::vector<std::string> check_need_filter(std::vector<std::shared_ptr<Alg_Node>> nodes,long long timestamp,bool is_real_time=false);

    void tick_forward_timestamp(std::vector<std::string> nodenames,long long timestamp);
    void tick_filter_timestamp(std::vector<std::string> nodenames,long long timestamp);
};

class Real_Cache{
    std::map<std::string,std::map<std::string,std::shared_ptr<InputOutput>>> cache_data;
    std::shared_mutex mutex;
public:
    void update(std::string module_name,std::map<std::string,std::shared_ptr<InputOutput>> data);
    std::map<std::string,std::shared_ptr<InputOutput>> get_data(std::string module_name);
    bool check(std::string module_name);
};


class CacheData_manager{
    std::map<std::string,std::map<std::string,std::shared_ptr<InputOutput>>> cache_data;
    std::map<std::string,std::map<std::string,int>> required_cnt;
public:
    std::shared_ptr<Real_Cache> real_cache=nullptr;

    void set_require_data(std::string module_name,std::string output_name);
    bool set_data(std::string module_name,std::string output_name,std::shared_ptr<InputOutput> data);
    std::shared_ptr<InputOutput> get_data(std::string module_name,std::string output_name);
    std::map<std::string,std::shared_ptr<InputOutput>> get_datas(std::shared_ptr<InputOutput_cfg> input_cfg);

    bool check_node_can_run(std::shared_ptr<InputOutput_cfg> inout_cfg);
};



class ChannelStatus{
public:
    long long last_timestamp=0;
    enum  Status{running,runtime_error,no_frame,no_config} status;

    void update(long long timestamp);
    Status get_status(long long timestamp);
};

class Alg_Engine{
    std::vector<std::shared_ptr<Alg_Node>> nodes;
    std::map<std::string,std::shared_ptr<Alg_Node>> nodes_map;

    std::map<std::string,std::shared_ptr<Tick_manager>> tick_managers;
    std::map<std::string,std::shared_ptr<Tick_manager>> tick_managers_real_time;

    std::map<std::string,std::shared_ptr<Real_Cache>> real_caches;

    std::map<std::string,std::map<std::string,int>> model_util_counter;

    std::set<std::string> global_input_names;


    std::map<int,std::shared_ptr<Device_Handle>> device_handles;

    std::map<std::string,std::string> has_config_channel;
    std::map<std::string,std::string> channel_cfgs;

    bool to_map(bool overwrite=false);
    std::vector<std::string> mark_required_module(std::vector<std::string>& module_names);
    std::vector<std::shared_ptr<Alg_Node>> gen_inference_queue(std::shared_ptr<Tick_manager>& tick_manager,long long timestamp,bool is_real_time=false);
    bool debug=false;
    std::shared_mutex mutex;

    std::shared_mutex module_edit_mutex;


    float add_instance_thres=0.7;
    float reduce_instance_thres=0.3;


public:
    std::map<std::string,ChannelStatus> channel_status;
    void set_add_reduce_instance_thres(float add_instance_thres,float reduce_instance_thres);

    Alg_Engine();
    virtual ~Alg_Engine();

    bool reload_module(std::string module_name);
    std::vector<std::string> get_module_names();
    void set_channel_cfg(std::string channel_name,std::string cfg_path);
    void enable_modules(std::vector<std::string> module_names,std::string channel_name="");
    void disable_modules(std::vector<std::string> module_names,std::string channel_name="");

    void get_enable_modules(std::string channel_name,std::vector<std::string>& module_names,std::vector<bool>& module_state);
    bool check_module_enable(std::string channel_name,std::shared_ptr<Alg_Node> module);
    void set_module_enable(std::string channel_name,std::shared_ptr<Alg_Node> module,bool enable);

    void set_device_ids(std::vector<int>& device_ids);

    void set_device_handles(std::vector<std::shared_ptr<Device_Handle>>& device_handles);
    

    bool load_module_from_libfile(std::string lib_path,std::string requirement_dir);
    bool load_module_from_libdir(std::string lib_dir,std::string requirement_dir);

    std::vector<Request_Model_instance_data> update_and_check_model_util(bool show_util=false);
    void update_model_instance_num(std::vector<Request_Model_instance_data>& data);

    std::vector<std::string> check_same_node_name();
    int check_need_forward(std::string channel_name, long long next_timestamp,bool is_real_time=false);

    std::vector<Node_require_check_result> check_node_publish();

    std::vector<Node_require_check_result> check_node_require();

    std::vector<Node_require_check_result> check_node_postprocess();

    void get_channel_status(std::string channel_name,std::vector<std::string>& out_channel_names,std::vector<uint8_t>& status,std::vector<std::string>& string);

    std::vector<std::shared_ptr<Publish_data>> forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input,long long timestamp, bool is_real_time=false);

    void print_module_time_summary();

};




#endif