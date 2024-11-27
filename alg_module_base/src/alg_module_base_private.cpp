#include "alg_module_base_private.h"
#include "error_type.h"
#include <filesystem>

Alg_Module_Base_private::Alg_Module_Base_private(std::string node_name){
    this->node_name=node_name;
};
Alg_Module_Base_private::~Alg_Module_Base_private(){

};

std::shared_ptr<Module_cfg_base> Alg_Module_Base_private::load_module_cfg_(std::string cfg_path){
    auto cfg = std::make_shared<Module_cfg_base>(node_name);
    int ret = cfg->from_file(cfg_path);
    if (ret < 0)
    {
        cfg.reset();
        std::cout<<"ERROR:\t module config file not exists or format error "<<cfg_path<<std::endl;
    }
    return cfg;

};
std::shared_ptr<Model_cfg_base> Alg_Module_Base_private::load_model_cfg_(std::string cfg_path){
    auto cfg = std::make_shared<Model_cfg_base>();
    if (cfg_path != "")
    {
        int ret = cfg->from_file(cfg_path);
        if (ret < 0)
        {

            std::cout<<"Warning:\t model config file not exists or format error "<<cfg_path<< "use default params"<<std::endl;
        }
    }
    return cfg;

};
std::shared_ptr<Channel_cfg_base> Alg_Module_Base_private::load_channel_cfg_(std::string channel_name,std::string cfg_path){
    auto channel_cfg = std::make_shared<Channel_cfg_base>(channel_name);
    int ret = channel_cfg->from_file(cfg_path);
    if (ret < 0)
    {
        channel_cfg.reset();
        std::cout<<"ERROR:\t channel "<<channel_name<<" config file not exists or format error "<<cfg_path<<std::endl;
    }
    else
    {
        //        this->channel_cfg[channel_name]=channel_cfg;
    }
    return channel_cfg;

};
std::shared_ptr<Channel_data_base> Alg_Module_Base_private::init_channal_data_(std::string channel_name){
    return std::make_shared<Channel_data_base>(channel_name);
};

bool Alg_Module_Base_private::load_model_(std::string model_path,std::string model_name,std::string model_cfg){
    if(model_cfg==""){
        if(this->get_model_cfg(model_name)==nullptr){
            std::shared_ptr<Model_cfg_base> cfg = this->load_model_cfg_(model_cfg);
            if (cfg != nullptr)
            {
                this->set_model_cfg(model_name, cfg);
            }
        }                
    }

    auto net = std::make_shared<Network_parallel>();
    net->load_model(model_path, model_name);
    std::cout<<"Info:\t load model "<<model_path<<" from file "<< model_name<<std::endl;

    auto lock = std::unique_lock(mutex);

    if (this->device_handle_cache.size() > 0)
    {
        net->set_device_handles(device_handle_cache);
    }
    networks[model_name] = net;
    return true;

};

void Alg_Module_Base_private::set_module_cfg(std::shared_ptr<Module_cfg_base> cfg){
    auto lock = std::unique_lock(this->mutex);
    this->module_cfg = cfg;

};
void Alg_Module_Base_private::set_model_cfg(std::string model_name,std::shared_ptr<Model_cfg_base> model_cfg_in){
    auto lock = std::unique_lock(this->mutex);
    this->model_cfg[model_name] = model_cfg_in;
    if(this->networks.find(model_name)!=this->networks.end()){
        this->networks[model_name]->set_input_shapes(model_cfg_in->input_shapes);
    }

};
void Alg_Module_Base_private::set_channel_cfg(std::shared_ptr<Channel_cfg_base> cfg){
    auto lock = std::unique_lock(this->mutex_channel_cfg);
    std::string channel_name = cfg->channel_name;
    this->channel_cfg[channel_name] = cfg;

};
void Alg_Module_Base_private::set_channal_data(std::string channel_name,std::shared_ptr<Channel_data_base> data){
    auto lock = std::unique_lock(this->mutex_channel_data);
    this->channel_data[channel_name] = data;

};


bool Alg_Module_Base_private::load_module_cfg(std::string cfg_path){
    auto ptr= this->load_module_cfg_(cfg_path);
    if(ptr==nullptr){
        throw Alg_Module_Exception("load module cfg error from "+cfg_path,this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }
    this->set_module_cfg(ptr);
    this->node_name=ptr->get_module_name();
    this->module_cfg->get_int("tick_interval",this->tick_interval_ms);  //优先使用配置文件中的运行帧率
    return true;
};
bool Alg_Module_Base_private::load_model_cfg(std::string cfg_path, std::string model_name){
    auto ptr= this->load_model_cfg_(cfg_path);
    if(ptr==nullptr){
        throw Alg_Module_Exception("load model cfg error from "+cfg_path,this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }
    this->set_model_cfg(model_name,ptr);

    return true;

};
bool Alg_Module_Base_private::load_channel_cfg(std::string channel_name, std::string cfg_path){
    auto ptr= this->load_channel_cfg_(channel_name,cfg_path);
    if(ptr==nullptr){
        throw Alg_Module_Exception("load channel cfg error from "+cfg_path,this->node_name,Alg_Module_Exception::Stage::load_channel);
        return false;
    }
    this->set_channel_cfg(ptr);
    return true;

};
bool Alg_Module_Base_private::init_channal_data(std::string channel_name){
    auto ptr= this->init_channal_data_(channel_name);
    if(ptr==nullptr){
        throw Alg_Module_Exception("init channel data error from "+channel_name,this->node_name,Alg_Module_Exception::Stage::init_channel_data);
        return false;
    }
    this->set_channal_data(channel_name,ptr);    
    return true;

};

bool Alg_Module_Base_private::load_model(std::string model_path, std::string model_name, std::string model_cfg){
    bool res=this->load_model_(model_path,model_name,model_cfg);
    if(res && this->model_cfg.find(model_name)!=this->model_cfg.end()){
        this->networks[model_name]->set_input_shapes(this->model_cfg[model_name]->input_shapes);
    }
    return res;
};

std::shared_ptr<Module_cfg_base> Alg_Module_Base_private::get_module_cfg(){
    return this->module_cfg;

};
std::shared_ptr<Model_cfg_base> Alg_Module_Base_private::get_model_cfg(std::string model_name){
    auto lock = std::shared_lock(this->mutex);
    if(this->model_cfg.find(model_name)==this->model_cfg.end()){
        return std::shared_ptr<Model_cfg_base>();
    }
    return this->model_cfg[model_name];

};
std::shared_ptr<Channel_cfg_base> Alg_Module_Base_private::get_channel_cfg(std::string channel_name){
    auto lock = std::shared_lock(this->mutex_channel_cfg);
    if(this->channel_cfg.find(channel_name)==this->channel_cfg.end()){
        return std::shared_ptr<Channel_cfg_base>();
    }
    return this->channel_cfg[channel_name];

};
std::shared_ptr<Channel_data_base> Alg_Module_Base_private::get_channal_data(std::string channel_name){
    auto lock = std::shared_lock(this->mutex_channel_data);
    if(this->channel_data.find(channel_name)==this->channel_data.end()){
        this->init_channal_data(channel_name);
        if(this->channel_data.find(channel_name)==this->channel_data.end()){

            return this->channel_data[channel_name];

        }
        else{
            return std::shared_ptr<Channel_data_base>();

        }
    }
    return this->channel_data[channel_name];

};
std::shared_ptr<Network_parallel> Alg_Module_Base_private::get_model_instance(std::string model_name)
{
    auto lock = std::shared_lock(this->mutex);
    std::shared_ptr<Network_parallel> res;
    if (this->networks.find(model_name) != this->networks.end())
    {
        res = this->networks[model_name];
    }
    return res;
};


bool Alg_Module_Base_private::get_device(std::shared_ptr<Device_Handle> &handle)
{
    if (this->device_ids.size() == 0)
    {
        return false;
    }
    int device_id = this->device_ids[this->tick_cnt++ % this->device_ids.size()];
    handle = this->device_handle_cache[device_id];
    return true;
};

void Alg_Module_Base_private::set_device_handles(std::vector<std::shared_ptr<Device_Handle>> device_handles)
{
    std::vector<int> device_ids;
    std::map<int, std::shared_ptr<Device_Handle>> new_device_handle_cache;
    for(auto iter=device_handles.begin();iter!=device_handles.end();iter++){
        
        int device_id=(*iter)->get_device_id();
        device_ids.push_back(device_id);
        new_device_handle_cache[device_id]=(*iter);
    }
    auto lock = std::unique_lock(this->mutex);
    
    this->device_ids = device_ids;
    this->device_handle_cache = new_device_handle_cache;
    for (auto iter = this->networks.begin(); iter != this->networks.end(); iter++)
    {
        iter->second->set_device_handles(device_handle_cache);
    }
    
}


void Alg_Module_Base_private::set_device_ids(std::vector<int> device_ids_new){
    std::vector<int> device_ids;
    std::map<int, std::shared_ptr<Device_Handle>> new_device_handle_cache;
    for (int i = 0; i < device_ids_new.size(); i++)
    {
        int id = device_ids_new[i];
        auto iter = this->device_handle_cache.find(id);
        if (iter == this->device_handle_cache.end())
        {
            Device_Handle* handle=get_device_handle(iter->first);
            new_device_handle_cache.insert(std::make_pair(id, handle));
            device_ids.push_back(id);
        }
        else
        {
            new_device_handle_cache.insert(std::make_pair(iter->first, iter->second));
            device_ids.push_back(id);
        }
    }
    auto lock = std::unique_lock(this->mutex);
    this->device_ids = device_ids;

    this->device_handle_cache = new_device_handle_cache;
    for (auto iter = this->networks.begin(); iter != this->networks.end(); iter++)
    {
        iter->second->set_device_handles(device_handle_cache);
    }
};

bool Alg_Module_Base_private::get_input_description(std::shared_ptr<InputOutput_cfg>& res){
    if(this->module_cfg==nullptr)
        return false;
    res=this->module_cfg->get_input_output_cfg();
    return true;
};

std::string Alg_Module_Base_private::get_module_name(){
    return this->node_name;
};
int Alg_Module_Base_private::get_module_tick_interval(){
    return this->tick_interval_ms;
};

bool Alg_Module_Base_private::reset_channal_data(std::string channel_name){
    auto lock=std::unique_lock(this->mutex_channel_data);
    if(this->channel_data.find(channel_name)==this->channel_data.end()){
        return true;
    }
    this->channel_data.erase(this->channel_data.find(channel_name));
    return true;
};

std::map<std::string,float> Alg_Module_Base_private::check_model_util(){
    std::map<std::string,float> result;
    for(auto iter=this->networks.begin();iter!=this->networks.end();iter++){
        result[iter->first]=iter->second->check_util();
    }
    return result;
};
bool Alg_Module_Base_private::increase_model_instane(std::string model_name,int device_id){
    if(this->networks.find(model_name)==this->networks.end())
        return false;

    return this->networks[model_name]->add_inference_instance(device_id);
};
bool Alg_Module_Base_private::reduce_model_instane(std::string model_name){
    if(this->networks.find(model_name)==this->networks.end())
        return false;

    return this->networks[model_name]->remove_last_inference_instance();

};
