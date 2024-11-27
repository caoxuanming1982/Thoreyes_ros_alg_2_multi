#include "alg_engine.h"
#include "error_type.h"

#include <boost/stacktrace.hpp>
#define BOOST_STACKTRACE_USE_ADDR2LINE
#define BOOST_STACKTRACE_USE_BACKTRACE


#include <exception>
#include <stdexcept>

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#define __WITH_TRY_CATCH__
const size_t kStacktraceDumpSize = 4096;

std::unordered_map<void*, const char*> stacktrace_dump_by_exc;
std::mutex mutex;

#ifdef DEBUG_TRACE
namespace __cxxabiv1 {
extern "C" {

extern void* __cxa_allocate_exception(size_t thrown_size) throw() {
  static thread_local bool already_in_allocate_exception = false;
  if (std::exchange(already_in_allocate_exception, true)) {  // for `bad_alloc`
    std::terminate();
  }

  typedef void* (*cxa_allocate_exception_t)(size_t);
  static auto orig_cxa_allocate_exception =
      (cxa_allocate_exception_t)dlsym(RTLD_NEXT, "__cxa_allocate_exception");

  static constexpr size_t kAlign = alignof(std::max_align_t);
  thrown_size = (thrown_size + kAlign - 1) & (~(kAlign - 1));  // round up

  void* user_obj_ptr =
      orig_cxa_allocate_exception(thrown_size + kStacktraceDumpSize);

  char* stacktrace_dump_ptr = ((char*)user_obj_ptr + thrown_size);

  // TODO: full dynamic serialization
  boost::stacktrace::safe_dump_to(1, stacktrace_dump_ptr, kStacktraceDumpSize);
  {
    std::lock_guard<std::mutex> lg{mutex};
    stacktrace_dump_by_exc[user_obj_ptr] = stacktrace_dump_ptr;
  }

  return already_in_allocate_exception = false, user_obj_ptr;
}

// TODO: Not called in libc++
// So the `stacktrace_dump_by_exc` is not cleared. That's not fatal
extern void __cxa_free_exception(void* thrown_object) throw() {
  static thread_local bool already_in_free_exception = false;
  if (std::exchange(already_in_free_exception, true)) {
    std::terminate();
  }

  typedef void (*cxa_free_exception_t)(void*);
  static auto orig_cxa_free_exception =
      (cxa_free_exception_t)dlsym(RTLD_NEXT, "__cxa_free_exception");
  orig_cxa_free_exception(thrown_object);

  {
    std::lock_guard<std::mutex> lg{mutex};
    stacktrace_dump_by_exc.erase(thrown_object);
  }

  already_in_free_exception = false;
}
}
}  // namespace __cxxabiv1

#endif

boost::stacktrace::stacktrace get_current_stacktrace_from_exception(){
    static const boost::stacktrace::stacktrace kEmpty{0, 0};  
#ifdef DEBUG_TRACE

    auto exc_ptr = std::current_exception();  
    void* exc_raw_ptr = *static_cast<void**>((void*)&exc_ptr);
    if (!exc_raw_ptr) {
        return kEmpty;
    }    
    const char* stacktrace_dump_ptr;
    {
        std::lock_guard<std::mutex> lg{mutex};
        auto it = stacktrace_dump_by_exc.find(exc_raw_ptr);
        if (it == stacktrace_dump_by_exc.end()) {
            return kEmpty;
        }
        stacktrace_dump_ptr = it->second;
    }   
    return boost::stacktrace::stacktrace::from_dump(stacktrace_dump_ptr,
                                                  kStacktraceDumpSize); 
#else
        return kEmpty;
#endif
}

void printStackTrace() {
    void* callstack[128];
    int i;
    int frames = backtrace(callstack, 128);
    char** strs = backtrace_symbols(callstack, frames);
    for (i = 0; i < frames; ++i) {
        printf("%s\n", strs[i]);
    }
    free(strs);
}

int is_dir(string path)
{
    if (path == "")
        return -1;
    if (access(path.c_str(), F_OK) == -1)
        return -1;
    DIR *dir = opendir(path.c_str());
    if (dir == NULL)
    {
        return 0;
    }
    else
    {
        closedir(dir);
    }
    return 1;
}

void GetLibFiles(std::string path, std::vector<std::string> &files_path)
{
    struct dirent *dirp;
    DIR *dir = opendir(path.c_str());
    while ((dirp = readdir(dir)) != nullptr)
    {
        if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
            continue;
        if (dirp->d_type == DT_REG)
        {

            if (std::string(dirp->d_name).find(".so") != std::string::npos)
            {
                files_path.push_back(path + "/" + dirp->d_name);
            }
        }
    }
    closedir(dir);
}

Alg_Node::Alg_Node(){

};
Alg_Node::~Alg_Node()
{
    if (module != nullptr)
    {
//        destory(module.get());
        module = nullptr;
    }
    if (handle != nullptr)
    {
        dlclose(handle);
        handle = nullptr;
    }
    create = nullptr;
    destory = nullptr;
};
bool Alg_Node::init(std::string requirement_root_dir_path)
{
    this->requirement_dir=requirement_root_dir_path;

    bool res = this->module->init_from_root_dir(requirement_root_dir_path);
    if (res == false)
        return res;
    this->filter_interval = this->module->get_module_tick_interval();
    this->tick_interval = this->module->get_module_tick_interval();
    this->module_name = this->module->get_module_name();
    this->time_counter.module_name=this->module_name;
    this->inout_cfg = this->module->get_module_cfg()->get_input_output_cfg();
    this->publish_cfg = this->module->get_module_cfg()->get_publish_cfg();
    this->post_process_cfg = this->module->get_module_cfg()->get_post_process_cfg();
    int real_time=0;
//    std::cout<<"init finish"<<std::endl;
    bool res1=this->module->get_module_cfg()->get_int("real_time",real_time);
    if(res1&&real_time>0){
        is_real_time=true;
    }

    int function_module=0;
    bool res2=this->module->get_module_cfg()->get_int("function_module",function_module);
    if(res2&&function_module>0){
        is_func_module=true;
    }
    if(is_func_module){
        Alg_Module_Base::set_function_module(module_name,this->module);
    }

    int need_cache=0;
    bool res3=this->module->get_module_cfg()->get_int("need_cache",need_cache);
    if(res3&&need_cache>0&&is_real_time){
        need_cache_for_unreal_time=true;
    }
    
    int no_event=0;
    bool res4=this->module->get_module_cfg()->get_int("no_event",no_event);
    if(res4&&no_event>0){
        no_filter=true;
    }

    
    return res;
};
bool Alg_Node::load(std::string lib_path)
{
    this->lib_path = lib_path;
    handle = dlopen(lib_path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!handle)
    {
        std::cout << "load alg module lib error" << std::endl;
        std::cout << dlerror() << std::endl;
        return false;
    }
    *(void **)(&create) = dlsym(handle, "create");
    const char *dlsym_error = dlerror();

    if (dlsym_error)
    {

        std::cout << "Cannot load symbol create: " << dlsym_error <<std::endl;

        return false;
    }
    *(void **)(&destory) = dlsym(handle, "destory");
    const char *dlsym_error1 = dlerror();

    if (dlsym_error1)
    {

        std::cout << "Cannot load symbol destory: " << dlsym_error1 <<std::endl;

        return false;
    }
    this->module = std::shared_ptr<Alg_Module_Base>(create());
    return true;
};

Inference_graph_node::Inference_graph_node(std::shared_ptr<Alg_Node> node)
{
    this->node = node;
};
Inference_graph::Inference_graph(){

};
void Inference_graph::insert_node(std::shared_ptr<Alg_Node> node)
{
    int next_idx = node2idx.size();
    nodes.push_back(std::make_shared<Inference_graph_node>(node));
    node2idx[node->module_name] = next_idx;
    nodes[nodes.size() - 1]->idx = next_idx;
};

void Inference_graph::make_edge()
{
    for (int i = 0; i < nodes.size(); i++)
    {
        auto node = nodes[i]->node;
        auto
            requires
        = node->inout_cfg->input_cfgs;
        for (auto iter = requires.begin(); iter != requires.end(); iter++)
        {
            if (iter->second.required_from_module == "")
            {
                continue;
            }
            nodes[i]->required_node_idx.insert(node2idx[iter->second.required_from_module]);
            nodes[node2idx[iter->second.required_from_module]]->next_node_idx.insert(i);
        }
    }
};

std::vector<std::shared_ptr<Alg_Node>> Inference_graph::gen_inference_queue()
{
    bool is_first = true;
    std::list<std::shared_ptr<Inference_graph_node>> temp;
    std::vector<std::shared_ptr<Alg_Node>> result;
    while (is_first || temp.size() > 0)
    {
        is_first = false;
        while (temp.size() > 0)
        {
            auto &ptr = temp.front();
            temp.pop_front();
            auto &next_node_idx = ptr->next_node_idx;
            result.push_back(ptr->node);
            for (auto iter = next_node_idx.begin(); iter != next_node_idx.end(); iter++)
            {
                nodes[*iter]->required_node_idx.erase(ptr->idx);
            }
        }
        for (int i = 0; i < nodes.size(); i++)
        {
            if (nodes[i]->required_node_idx.size() == 0 && nodes[i]->is_infered == false)
            {
                nodes[i]->is_infered = true;
                temp.push_back(nodes[i]);
            }
        }
    }
    return result;
};
Tick_manager::Tick_manager(std::string channel_name)
{
    this->channel_name = channel_name;
};
Tick_manager::~Tick_manager(){};

std::vector<std::string> Tick_manager::check_need_forward(std::vector<std::shared_ptr<Alg_Node>> nodes, long long timestamp,bool is_real_time)
{
    std::vector<std::string> res;
    for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
    {
        if(is_real_time){

            if((*iter)->is_real_time==false){
                continue;
            }
        }
        else{

            if((*iter)->is_real_time){
                continue;
            }

        }

        std::string module_name = (*iter)->module_name;
        if (last_forward_timestamp.find(module_name) == last_forward_timestamp.end())
        {
            last_forward_timestamp[module_name] = 0;
        }
        bool enable = (*iter)->enable;
        if ((*iter)->channel_enable.find(channel_name) != (*iter)->channel_enable.end())
        {
            enable = (*iter)->channel_enable[channel_name];
        }

        if ((*iter)->tick_interval > 0 && timestamp > last_forward_timestamp[module_name] && timestamp - last_forward_timestamp[module_name] >= (*iter)->tick_interval - 30 && enable)
        {
            res.push_back(module_name);
        }
    }
    return res;
};

std::vector<std::string> Tick_manager::check_need_filter(std::vector<std::shared_ptr<Alg_Node>> nodes, long long timestamp,bool is_real_time)
{
    std::vector<std::string> res;
    for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
    {
        if(is_real_time){

            if((*iter)->is_real_time==false){
                continue;
            }
        }
        else{

            if((*iter)->is_real_time){
                continue;
            }

        }

        std::string module_name = (*iter)->module_name;
        if (last_filter_timestamp.find(module_name) == last_filter_timestamp.end())
        {
            last_filter_timestamp[module_name] = 0;
        }
        bool enable = (*iter)->enable;
        if ((*iter)->channel_enable.find(channel_name) != (*iter)->channel_enable.end())
        {
            enable = (*iter)->channel_enable[channel_name];
        }
        //        cout<<"check enable "<< (*iter)->module_name<<" channel "<<channel_name<< " "<<enable<<std::endl;
        if ((*iter)->tick_interval > 0 && timestamp > last_filter_timestamp[module_name] && timestamp - last_filter_timestamp[module_name] >= (*iter)->tick_interval - 30 && enable)
        {
            res.push_back(module_name);
        }
    }
    return res;
};

void Tick_manager::tick_forward_timestamp(std::vector<std::string> nodenames, long long timestamp)
{
    for (auto iter = nodenames.begin(); iter != nodenames.end(); iter++)
    {
        this->last_forward_timestamp[*iter] = timestamp;
    }
};
void Tick_manager::tick_filter_timestamp(std::vector<std::string> nodenames, long long timestamp)
{
    for (auto iter = nodenames.begin(); iter != nodenames.end(); iter++)
    {
        this->last_filter_timestamp[*iter] = timestamp;
    }
};

void CacheData_manager::set_require_data(std::string module_name, std::string output_name)
{
    if (required_cnt.find(module_name) == required_cnt.end())
    {
        required_cnt[module_name] = std::map<std::string, int>();
    }
    if (required_cnt[module_name].find(output_name) == required_cnt[module_name].end())
    {
        required_cnt[module_name][output_name] = 1;
    }
    else
    {
        required_cnt[module_name][output_name] += 1;
    }
};
bool CacheData_manager::set_data(std::string module_name, std::string output_name, std::shared_ptr<InputOutput> data)
{
    if (cache_data.find(module_name) == cache_data.end())
    {
        cache_data[module_name] = std::map<std::string, std::shared_ptr<InputOutput>>();
    }
    if (cache_data[module_name].find(output_name) == cache_data[module_name].end())
    {
        cache_data[module_name][output_name] = data;
    }
    return true;
}
std::shared_ptr<InputOutput> CacheData_manager::get_data(std::string module_name, std::string output_name)
{
    std::shared_ptr<InputOutput> result;
    if (cache_data.find(module_name) != cache_data.end())
    {
        if(real_cache->check(module_name)){
            std::map<std::string,std::shared_ptr<InputOutput>> temp= real_cache->get_data(module_name);
            if(temp.find(output_name) != temp.end()){
                result=temp[output_name];
                required_cnt[module_name][output_name] -= 1;
                return result;
            }            
        }

        if (cache_data[module_name].find(output_name) != cache_data[module_name].end())
        {
            result = cache_data[module_name][output_name];
            required_cnt[module_name][output_name] -= 1;
            if (required_cnt[module_name][output_name] <= 0)
            {
                required_cnt[module_name].erase(output_name);
                cache_data[module_name].erase(output_name);
            }
            if (required_cnt[module_name].size() <= 0)
            {
                required_cnt.erase(module_name);
                cache_data.erase(module_name);
            }
        }
    }

    return result;
}
std::map<std::string, std::shared_ptr<InputOutput>> CacheData_manager::get_datas(std::shared_ptr<InputOutput_cfg> input_cfg)
{
    std::map<std::string, std::shared_ptr<InputOutput>> result;
    for (auto iter = input_cfg->input_cfgs.begin(); iter != input_cfg->input_cfgs.end(); iter++)
    {
        if (iter->second.required_from_module == "")
        {
            result[iter->first] = get_data(iter->second.required_from_module, iter->first);
        }
        else
        {
            result[iter->first] = get_data(iter->second.required_from_module, iter->second.required_from_module_output_name);
        }
    }
    return result;
};

bool CacheData_manager::check_node_can_run(std::shared_ptr<InputOutput_cfg> inout_cfg)
{
    for (auto iter = inout_cfg->input_cfgs.end(); iter != inout_cfg->input_cfgs.end(); iter++)
    {
        if (iter->second.required_from_module == "")
        {
            continue;
        }
        if(real_cache->check(iter->second.required_from_module)){
            return true;
        }

        if (cache_data.find(iter->second.required_from_module) == cache_data.end())
        {
            return false;
        }
        auto &module_outputs = cache_data[iter->second.required_from_module];

        if (module_outputs.find(iter->second.required_from_module_output_name) == module_outputs.end())
        {
            return false;
        }
    }
    return true;
};

void Real_Cache::update(std::string module_name,std::map<std::string,std::shared_ptr<InputOutput>> data){
    std::shared_lock<std::shared_mutex> lock(mutex);
    cache_data[module_name] = data;
};
std::map<std::string,std::shared_ptr<InputOutput>> Real_Cache::get_data(std::string module_name){

    std::shared_lock<std::shared_mutex> lock(mutex);
    return cache_data[module_name];
};
bool Real_Cache::check(std::string module_name){
    if(cache_data.find(module_name)==cache_data.end()){
        return false;
    }
    return true;
};


bool Alg_Engine::to_map(bool overwrite)
{
    if (overwrite || nodes_map.size() != nodes.size())
    {
        for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
        {
            nodes_map[(*iter)->module_name] = *iter;
        }
    }
    return false;
};
std::vector<std::string> Alg_Engine::mark_required_module(std::vector<std::string> &module_names)
{

    std::list<std::string> input(module_names.begin(), module_names.end());
    std::set<std::string> result;
    while (input.size() > 0)
    {
        std::string module_name = input.front();
        input.pop_front();
        result.insert(module_name);
        auto
            requires
        = nodes_map[module_name]->inout_cfg->input_cfgs;
        for (auto iter = requires.begin(); iter != requires.end(); iter++)
        {
            if (iter->second.required_from_module != "" && result.find(iter->second.required_from_module) == result.end())
                input.push_back(iter->second.required_from_module);
        }
    }
    return std::vector<std::string>(result.begin(), result.end());
};
std::vector<std::shared_ptr<Alg_Node>> Alg_Engine::gen_inference_queue(std::shared_ptr<Tick_manager> &tick_manager, long long timestamp,bool is_real_time)
{
    std::vector<std::string> inference_module_names = tick_manager->check_need_forward(this->nodes, timestamp,is_real_time);
    std::vector<std::string> filter_module_names = tick_manager->check_need_filter(this->nodes, timestamp,is_real_time);
    std::vector<std::string> need_run_module_names;
    if (inference_module_names.size() == 0)
        need_run_module_names = filter_module_names;
    else if (filter_module_names.size() == 0)
        need_run_module_names = inference_module_names;
    else
    {
        need_run_module_names.resize(inference_module_names.size() + filter_module_names.size());
        auto iter_end = std::set_union(inference_module_names.begin(), inference_module_names.end(), filter_module_names.begin(), filter_module_names.end(), need_run_module_names.begin());
        need_run_module_names.resize(iter_end - need_run_module_names.begin());
    }

    need_run_module_names = this->mark_required_module(need_run_module_names);
    Inference_graph graph;
    for (int i = 0; i < need_run_module_names.size(); i++)
    {
        graph.insert_node(nodes_map[need_run_module_names[i]]);
    }
    graph.make_edge();
    return graph.gen_inference_queue();
};

void ChannelStatus::update(long long timestamp)
{
    this->last_timestamp = timestamp;
    if (this->status != ChannelStatus::Status::runtime_error)
    {
        this->status = ChannelStatus::Status::running;
    }
};

ChannelStatus::Status ChannelStatus::get_status(long long timestamp)
{
    if (timestamp - this->last_timestamp > 30000)
    {
        if (this->status != ChannelStatus::Status::runtime_error)
            this->status = ChannelStatus::Status::no_frame;
    }
    return this->status;
};

Alg_Engine::Alg_Engine()
{
    global_input_names.insert("image");
    global_input_names.insert("pointcloud");
    global_input_names.insert("radar_object");
    global_input_names.insert("timestamp");
};
Alg_Engine::~Alg_Engine(){

};
void Alg_Engine::set_channel_cfg(std::string channel_name, std::string cfg_path)
{
    for (int i = 0; i < nodes.size(); i++)
    {
        std::cout << "set channel " << channel_name << " module " << nodes[i]->module_name << " cfg" << std::endl;
        nodes[i]->module->load_channel_cfg(channel_name, cfg_path);
        nodes[i]->module->reset_channal_data(channel_name);
        nodes[i]->module->init_channal_data(channel_name);

        channel_cfgs[channel_name]=cfg_path;
    }
    this->has_config_channel[channel_name] = cfg_path;
};
void Alg_Engine::set_device_handles(std::vector<std::shared_ptr<Device_Handle>>& device_handles){
    std::map<int, std::shared_ptr<Device_Handle>> new_device_handles;
    for (int i = 0; i < device_handles.size(); i++)
    {
        int device_id = device_handles[i]->get_device_id();
        new_device_handles[device_id] = device_handles[i];
    }
    this->device_handles = new_device_handles;
    for (int i = 0; i < nodes.size(); i++)
    {
        nodes[i]->module->set_device_handles(device_handles);
    }
};

void Alg_Engine::set_device_ids(std::vector<int> &device_ids)
{
    std::map<int, std::shared_ptr<Device_Handle>> new_device_handles;
    std::vector<std::shared_ptr<Device_Handle>> handles;

    for (int i = 0; i < device_ids.size(); i++)
    {
        if (this->device_handles.find(device_ids[i]) == this->device_handles.end())
        {
            std::shared_ptr<Device_Handle> handle=std::shared_ptr<Device_Handle>(get_device_handle(device_ids[i]));
            handles.push_back(handle);
            new_device_handles[device_ids[i]] = handle;
        }
        else
        {
            new_device_handles[device_ids[i]] = this->device_handles[device_ids[i]];
            handles.push_back(this->device_handles[device_ids[i]]);
        }
    }
    this->device_handles = new_device_handles;
    for (int i = 0; i < nodes.size(); i++)
    {
        nodes[i]->module->set_device_handles(handles);
    }
};

bool Alg_Engine::load_module_from_libfile(std::string lib_path, std::string requirement_dir)
{
    auto node = std::make_shared<Alg_Node>();
    bool res = node->load(lib_path);
    if (res)
    {
        res = node->init(requirement_dir);
    }
    if (res)
    {
        if (this->device_handles.size() > 0)
        {
            std::vector<std::shared_ptr<Device_Handle>> handles;


            for (auto iter = this->device_handles.begin(); iter != device_handles.end(); iter++)
            {
                handles.push_back(iter->second);
            }
            node->module->set_device_handles(handles);
        }
        this->nodes.push_back(node);
    }
    return res;
};
bool Alg_Engine::load_module_from_libdir(std::string lib_dir, std::string requirement_dir)
{
    if (is_dir(lib_dir) > 0)
    {
        std::vector<std::string> lib_paths;
        GetLibFiles(lib_dir, lib_paths);
        for (int i = 0; i < lib_paths.size(); i++)
        {
            bool res = this->load_module_from_libfile(lib_paths[i], requirement_dir);
            if (res == false)
            {
                std::cout << "load alg module error from " << lib_paths[i] << std::endl;
                return false;
            }
        }
        std::vector<std::string> names=Alg_Module_Base::get_function_module_name();
        std::cout<<"function modules:"<<std::endl;
        for(int i=0;i<names.size();i++)
        {
            std::cout<<"\t"<<names[i]<<std::endl;
        }
        std::cout<<std::endl;
        return true;
    }
    else
    {
        std::cout << "alg module dir not exists " << is_dir(lib_dir) << std::endl;
        return false;
    }
};


void Alg_Engine::set_add_reduce_instance_thres(float add_instance_thres,float reduce_instance_thres){
    this->add_instance_thres=add_instance_thres;
    this->reduce_instance_thres=reduce_instance_thres;


};

std::vector<Request_Model_instance_data> Alg_Engine::update_and_check_model_util(bool show_util)
{
    std::vector<Request_Model_instance_data> result;
    if(show_util){
            std::cout << "=====================model utils=====================" << std::endl;
    }
    std::shared_lock<std::shared_mutex> lock(module_edit_mutex);;
    for (int i = 0; i < nodes.size(); i++)
    {
        std::string module_name = nodes[i]->module_name;
        auto model_utils = nodes[i]->module->check_model_util();
        if (model_util_counter.find(module_name) == model_util_counter.end())
        {
            model_util_counter[module_name] = std::map<std::string, int>();
        }
        auto &temp = model_util_counter[module_name];
        for (auto iter = model_utils.begin(); iter != model_utils.end(); iter++)
        {
            if (temp.find(iter->first) == temp.end())
            {
                temp[iter->first] = 0;
                Request_Model_instance_data item;
                item.cfg = nodes[i]->module->get_model_cfg(iter->first);
                item.model_name = iter->first;
                item.module_name = module_name;
                item.has_ins=iter->second>10?false:true;
                result.push_back(item);
                continue;
            }

            if (iter->second > add_instance_thres)
//            if (iter->second > 0.5)
            {
                if (temp[iter->first] < 0)
                    temp[iter->first] = 0;
                temp[iter->first] += 1;
            }
            else if (iter->second < reduce_instance_thres)
//            else if (iter->second < 0.2)
            {
                if (temp[iter->first] > 0)
                    temp[iter->first] = 0;
                temp[iter->first] -= 1;
            }
            else
            {
                temp[iter->first] *= 0.8;
            }
            if(show_util){
                std::cout << iter->first << " util " << iter->second<<"\t"<<temp[iter->first] << std::endl;
            }

            if (temp[iter->first] > 8)
            {
                Request_Model_instance_data item;
                item.cfg = nodes[i]->module->get_model_cfg(iter->first);
                item.model_name = iter->first;
                item.module_name = module_name;
                item.has_ins=iter->second>10?false:true;
                result.push_back(item);
                temp[iter->first] = 0;
            }
            else if (temp[iter->first] < -8)
            {
                nodes[i]->module->reduce_model_instane(iter->first);
                temp[iter->first] = 0;
            }
        }
    }
    if(show_util){
            std::cout << "=====================end model utils=====================" << std::endl;
    }

    return result;
};
void Alg_Engine::update_model_instance_num(std::vector<Request_Model_instance_data> &data)
{
    std::shared_lock<std::shared_mutex> lock(module_edit_mutex);;
    to_map();
    for (int i = 0; i < data.size(); i++)
    {
        if (data[i].result_device_id >= 0)
        {
            nodes_map[data[i].module_name]->module->increase_model_instane(data[i].model_name, data[i].result_device_id);
        }
    }
};

std::vector<std::string> Alg_Engine::check_same_node_name()
{
    std::map<std::string, int> name_cnt;
    for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
    {
        if (name_cnt.find((*iter)->module_name) == name_cnt.end())
        {
            name_cnt[(*iter)->module_name] = 1;
        }
        else
        {
            name_cnt[(*iter)->module_name] += 1;
        }
    }
    std::vector<std::string> res;
    for (auto iter = name_cnt.begin(); iter != name_cnt.end(); iter++)
    {
        if (iter->second > 1)
            res.push_back(iter->first);
    }
    return res;
};

std::vector<Node_require_check_result> Alg_Engine::check_node_publish()
{
    to_map();
    std::vector<Node_require_check_result> result;
    for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
    {
        auto &output_cfg = (*iter)->inout_cfg->output_cfgs;
        auto &publish_cfg1 = (*iter)->publish_cfg->filter_publish_cfg;
        auto &publish_cfg2 = (*iter)->publish_cfg->raw_publish_cfg;
        for (int i = 0; i < publish_cfg1.size(); i++)
        {
            if (publish_cfg1[i].need_publish)
            {
                if (output_cfg.find(publish_cfg1[i].output_result_name) == output_cfg.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = publish_cfg1[i].output_result_name;
                    item.error_msg = "publish on a invalid output";
                    result.push_back(item);
                }
            }
        }
        for (int i = 0; i < publish_cfg2.size(); i++)
        {
            if (publish_cfg2[i].need_publish)
            {
                if (output_cfg.find(publish_cfg2[i].output_result_name) == output_cfg.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = publish_cfg2[i].output_result_name;
                    item.error_msg = "publish on a invalid output";
                    result.push_back(item);
                }
            }
        }
    }
    return result;
}
std::vector<Node_require_check_result> Alg_Engine::check_node_postprocess()
{
    to_map();
    std::vector<Node_require_check_result> result;
    for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
    {
        auto &cfgs = (*iter)->post_process_cfg->post_process_cfgs;
        auto &output_cfg = (*iter)->inout_cfg->output_cfgs;
        for (auto iter1 = cfgs.begin(); iter1 != cfgs.end(); iter1++)
        {
            std::string module_name = iter1->module_name;
            if (this->nodes_map.find(module_name) == this->nodes_map.end())
            {
                Node_require_check_result item;
                item.module_name = (*iter)->module_name;
                item.param_name = "post process module name";
                item.error_msg = module_name + " not in current module set";
                result.push_back(item);
                continue;
            }
            auto &next_node_input = this->nodes_map[module_name]->inout_cfg->input_cfgs;
            auto &next_node_output = this->nodes_map[module_name]->inout_cfg->output_cfgs;

            for (auto iter2 = iter1->map_input.begin(); iter2 != iter1->map_input.end(); iter2++)
            {
                if (output_cfg.find(iter2->second) == output_cfg.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = "post process module input";
                    item.error_msg = iter2->second + " not in module <" + module_name + "> output";
                    result.push_back(item);
                }
                if (next_node_input.find(iter2->first) == next_node_input.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = "post process module input";
                    item.error_msg = iter2->first + " not in next module <" + module_name + "> output";
                    result.push_back(item);
                }
            }

            for (auto iter2 = iter1->map_output.begin(); iter2 != iter1->map_output.end(); iter2++)
            {
                if (output_cfg.find(iter2->second) == output_cfg.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = "post process module output";
                    item.error_msg = iter2->second + " not in module <" + module_name + "> output";
                    result.push_back(item);
                }
                if (next_node_output.find(iter2->first) == next_node_output.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = "post process module output";
                    item.error_msg = iter2->first + " not in next module <" + module_name + "> output";
                    result.push_back(item);
                }
            }
        }
    }
    return result;
}
std::vector<Node_require_check_result> Alg_Engine::check_node_require()
{
    to_map();
    std::vector<Node_require_check_result> result;
    for (auto iter = nodes.begin(); iter != nodes.end(); iter++)
    {
        auto &input_cfg = (*iter)->inout_cfg->input_cfgs;
        for (auto iter1 = input_cfg.begin(); iter1 != input_cfg.end(); iter1++)
        {
            std::string require_module_name = iter1->second.required_from_module;
            std::string require_param_name = iter1->second.required_from_module_output_name;
            if (require_module_name == "")
            {
                if (global_input_names.find(iter1->first) == global_input_names.end() && (*iter)->filter_interval > 0)
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = iter1->first;
                    item.error_msg = "required global input not exist";
                    result.push_back(item);
                    continue;
                }
            }
            else if (nodes_map.find(require_module_name) == nodes_map.end())
            {
                Node_require_check_result item;
                item.module_name = (*iter)->module_name;
                item.param_name = iter1->first;
                item.error_msg = "required module not exists";
                result.push_back(item);
                continue;
            }
            else
            {
                auto output_cfg = nodes_map[require_module_name]->inout_cfg->output_cfgs;
                if (output_cfg.find(require_param_name) == output_cfg.end())
                {
                    Node_require_check_result item;
                    item.module_name = (*iter)->module_name;
                    item.param_name = iter1->first;
                    item.error_msg = "required param not exists";
                    result.push_back(item);
                    continue;
                }
                else
                {
                    if (iter1->second.data_type != output_cfg[require_param_name])
                    {
                        Node_require_check_result item;
                        item.module_name = (*iter)->module_name;
                        item.param_name = iter1->first;
                        item.error_msg = "required param type miss match";
                        result.push_back(item);
                        continue;
                    }
                }
            }
        }
    }
    return result;
};
int Alg_Engine::check_need_forward(std::string channel_name, long long next_timestamp,bool is_real_time)
{
    auto lock = std::shared_lock(this->mutex);
    if (tick_managers.find(channel_name) == tick_managers.end())
    {
        tick_managers[channel_name] = std::make_shared<Tick_manager>(channel_name);
    }
    std::shared_lock<std::shared_mutex> edit_lock(module_edit_mutex);;
    std::vector<std::string> filter_module_names = tick_managers[channel_name]->check_need_filter(this->nodes, next_timestamp,is_real_time);
    return filter_module_names.size();
};

bool Alg_Engine::reload_module(std::string module_name){
    if(this->nodes_map.find(module_name)!=this->nodes_map.end()){
        std::string lib_path=this->nodes_map[module_name]->lib_path;
        std::string requirement_dir=this->nodes_map[module_name]->requirement_dir;
        auto node = std::make_shared<Alg_Node>();
        bool res = node->load(lib_path);
        if (res)
        {
            res = node->init(requirement_dir);
        }
        if (res)
        {
            if (this->device_handles.size() > 0)
            {
            std::vector<std::shared_ptr<Device_Handle>> handles;
            for (auto iter = this->device_handles.begin(); iter != device_handles.end(); iter++)
            {
                handles.push_back(iter->second);
            }
                node->module->set_device_handles(handles);

            }
            node->channel_enable=this->nodes_map[module_name]->channel_enable;
            node->enable=this->nodes_map[module_name]->enable;
            for(auto iter=this->channel_cfgs.begin();iter!=channel_cfgs.end();iter++){
                node->module->load_channel_cfg(iter->first, iter->second);
                node->module->reset_channal_data(iter->first);
                node->module->init_channal_data(iter->first);
            }            
            auto old_module=this->nodes_map[module_name];
            for(int i=0;i<this->nodes.size();i++){
                if(this->nodes[i]==old_module){
                    std::unique_lock<std::shared_mutex> lock(module_edit_mutex);;
                    this->nodes[i]=node;
                    this->nodes_map[module_name]=node;
                    return true;
                }

            }

       }

    }
    return false;

};


void debug_show(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> input, std::map<std::string, std::shared_ptr<InputOutput>> filtered_output)
{
    static int idx = 0;
    static std::shared_mutex mutex;
    auto lock = std::unique_lock(mutex);
    cv::Mat image;
    if (input["image"]->data_type == InputOutput::Type::Image_t)
    {
        image = input["image"]->data.image->get_image();
    }
    for (auto it = filtered_output.begin(); it != filtered_output.end(); it++)
    {

        if (it->second->data_type == InputOutput::Type::Result_Detect_t)
        {
            auto &result = it->second->data.detect;
            for (int i = 0; i < result.size(); i++)
            {
                auto &images = result[i].res_images;
                for (auto iter = images.begin(); iter != images.end(); iter++)
                {
                    std::string file_name = Num2string<int>(idx) + "_" + iter->first + ".png";
                    cv::imwrite("/home/vensin/worksapce/temp_result/" + file_name, iter->second);
                    std::string file_name1 = Num2string<int>(idx) + "_" + iter->first + "_orig.png";
                    cv::imwrite("/home/vensin/worksapce/temp_result/" + file_name1, image);
                    idx += 1;
                }
            }
        }
        else if (it->second->data_type == InputOutput::Type::Result_Detect_license_t)
        {
            auto &result = it->second->data.detect_license;
            for (int i = 0; i < result.size(); i++)
            {
                auto &images = result[i].res_images;
                for (auto iter = images.begin(); iter != images.end(); iter++)
                {
                    std::string file_name = Num2string<int>(idx) + "_" + iter->first + ".png";
                    cv::imwrite("/home/vensin/worksapce/temp_result/" + file_name, iter->second);
                    std::string file_name1 = Num2string<int>(idx) + "_" + iter->first + "_orig.png";
                    cv::imwrite("/home/vensin/worksapce/temp_result/" + file_name1, image);
                    idx += 1;
                }
            }
        }
    }
};

std::vector<std::shared_ptr<Publish_data>> Alg_Engine::forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, long long timestamp,bool is_real_time)
{
    if (real_caches.find(channel_name) == real_caches.end())
    {
        real_caches[channel_name] = std::make_shared<Real_Cache>();
    }


    if (channel_status.find(channel_name) == channel_status.end())
    {
        channel_status[channel_name] = ChannelStatus();
    }
    if (has_config_channel.find(channel_name) == has_config_channel.end())
    {
        if (this->debug)
        {
            std::cout << "channel " << channel_name << " has no channel cfg" << std::endl;
        }
        channel_status[channel_name].status = ChannelStatus::Status::no_config;
        return std::vector<std::shared_ptr<Publish_data>>();
    }


    std::shared_lock<std::shared_mutex> edit_lock(module_edit_mutex);;
    auto lock = std::unique_lock(this->mutex);
    std::vector<std::shared_ptr<Publish_data>> publish_results_all;

    CacheData_manager cache_manage;
    cache_manage.real_cache=real_caches[channel_name];
    if (tick_managers.find(channel_name) == tick_managers.end())
    {
        tick_managers[channel_name] = std::make_shared<Tick_manager>(channel_name);
    }
    std::vector<std::string> filter_module_names = tick_managers[channel_name]->check_need_filter(this->nodes, timestamp,is_real_time);
    std::vector<std::shared_ptr<Alg_Node>> tasks = gen_inference_queue(tick_managers[channel_name], timestamp,is_real_time);
/*
    for(int i=0;i<filter_module_names.size();i++)
    {
        std::cout<<filter_module_names[i]<<"\t";
    }
    std::cout<<std::endl;

    for(int i=0;i<tasks.size();i++)
    {
        std::cout<<tasks[i]->module_name<<"\t";
    }
    std::cout<<std::endl;

*/
    for (auto iter = input.begin(); iter != input.end(); iter++)
    {
        cache_manage.set_data("", iter->first, iter->second);
    }
    std::vector<std::string> inference_module_names;
    for (auto iter = tasks.begin(); iter != tasks.end(); iter++)
    {
        auto &input_cfg = (*iter)->inout_cfg->input_cfgs;
        for (auto iter1 = input_cfg.begin(); iter1 != input_cfg.end(); iter1++)
        {
            if (iter1->second.required_from_module == "")
            {
                cache_manage.set_require_data(iter1->second.required_from_module, iter1->first);
            }
            else
            {
                cache_manage.set_require_data(iter1->second.required_from_module, iter1->second.required_from_module_output_name);
            }
        }
        inference_module_names.push_back((*iter)->module_name);
    }

    tick_managers[channel_name]->tick_filter_timestamp(filter_module_names, timestamp);
    tick_managers[channel_name]->tick_forward_timestamp(inference_module_names, timestamp);
    lock.unlock();
    channel_status[channel_name].update(timestamp);
    channel_status[channel_name].status = ChannelStatus::Status::running;

    std::set<std::string> filter_module_names_set(filter_module_names.begin(), filter_module_names.end());
    std::vector<std::future<std::vector<std::shared_ptr<Publish_data>>>> futures;
    std::vector<std::string> futures_for_node;


    for (auto iter = tasks.begin(); iter != tasks.end(); iter++)
    {
        if(is_real_time==false){
            if(cache_manage.real_cache->check((*iter)->module_name)){
                continue;
            }
        }

        if (cache_manage.check_node_can_run((*iter)->inout_cfg))
        {
            std::map<std::string, std::shared_ptr<InputOutput>> module_input = cache_manage.get_datas((*iter)->inout_cfg);
            std::map<std::string, std::shared_ptr<InputOutput>> module_output;
            //                debug_show(channel_name,module_input,module_output);
            if ((*iter)->module->get_channal_data(channel_name) == nullptr)
            {
                (*iter)->module->init_channal_data(channel_name);
            }

#ifdef __WITH_TRY_CATCH__
            try
            {
#endif
                (*iter)->time_counter.start_inference(channel_name,get_time());
                (*iter)->module->forward(channel_name, module_input, module_output);
                (*iter)->time_counter.end_inference(channel_name,get_time());
#ifdef __WITH_TRY_CATCH__
            }
            catch (Alg_Module_Exception &e)
            {
                auto st= get_current_stacktrace_from_exception();
                e.set_trace(st);
                throw e;
            }
            catch (std::runtime_error &e)
            {
                auto st= get_current_stacktrace_from_exception();
                Alg_Module_Exception except_data(e.what(), (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::inference, channel_name);
                except_data.set_trace(st);
                throw except_data;
            }
            catch (cv::Exception &e)
            {                
                auto st= get_current_stacktrace_from_exception();
                Alg_Module_Exception except_data(e.msg+"\n"+e.func+"\n"+e.file+"\n"+e.err, (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::inference, channel_name);
                except_data.set_trace(st);
                throw except_data;
            }
            catch (...)
            {                
                auto st= get_current_stacktrace_from_exception();
                Alg_Module_Exception except_data("unknown runtime error", (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::inference, channel_name);
                except_data.set_trace(st);
                throw except_data;
            }
#endif
            if ((*iter)->post_process_cfg->post_process_cfgs.size() > 0)
            {
                for (auto iter1 = (*iter)->post_process_cfg->post_process_cfgs.begin(); iter1 != (*iter)->post_process_cfg->post_process_cfgs.end(); iter1++)
                {

                    std::map<std::string, std::shared_ptr<InputOutput>> module_input_t = module_input;
                    std::shared_ptr<Alg_Node> next_node = this->nodes_map[iter1->module_name];
                    if (next_node->module->get_channal_data(channel_name) == nullptr)
                    {
                        next_node->module->init_channal_data(channel_name);
                    }

                    for (auto iter2 = iter1->map_input.begin(); iter2 != iter1->map_input.end(); iter2++)
                    {
                        module_input_t[iter2->first] = module_output[iter2->second];
                    }
                    std::map<std::string, std::shared_ptr<InputOutput>> module_output_t;
#ifdef __WITH_TRY_CATCH__
                    try
                    {
#endif
                        next_node->time_counter.start_inference(channel_name,get_time());
                        next_node->module->forward(channel_name, module_input_t, module_output_t);
                        next_node->time_counter.end_inference(channel_name,get_time());
#ifdef __WITH_TRY_CATCH__

                    }
                    catch (Alg_Module_Exception &e)
                    {
                        e.set_trace(e.st);
                        throw e;
                    }
                    catch (std::runtime_error &e)
                    {
                        auto st= get_current_stacktrace_from_exception();
                        Alg_Module_Exception except_data(e.what(), (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::inference, channel_name);
                        except_data.set_trace(st);
                        throw except_data;
                    }
                    catch (cv::Exception &e)
                    {                
                        auto st= get_current_stacktrace_from_exception();
                        Alg_Module_Exception except_data(e.msg+"\n"+e.func+"\n"+e.file+"\n"+e.err, (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::inference, channel_name);
                        except_data.set_trace(st);
                        throw except_data;
                    }
                    catch (...)
                    {
                        auto st= get_current_stacktrace_from_exception();
                        Alg_Module_Exception except_data("unknown runtime error", (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::inference, channel_name);
                        except_data.set_trace(st);
                        throw except_data;
                    }
#endif
                    for (auto iter2 = iter1->map_output.begin(); iter2 != iter1->map_output.end(); iter2++)
                    {
                        module_output[iter2->second] = module_output_t[iter2->first];
                    }
                }
            }

                    if((*iter)->need_cache_for_unreal_time){
                        cache_manage.real_cache->update((*iter)->module_name,module_output);
                        if(check_module_enable(channel_name,*iter)==false){
                            set_module_enable(channel_name,*iter,true);
                        }
                }
        

            for (auto iter1 = module_output.begin(); iter1 != module_output.end(); iter1++)
            {
                cache_manage.set_data((*iter)->module_name, iter1->first, iter1->second);
            }
//        futures_for_node.push_back((*iter)->module_name);
  //      futures.push_back(std::async(std::launch::async, [iter, filter_module_names_set, module_input, module_output, channel_name]() mutable{
            std::vector<std::shared_ptr<Publish_data>> publish_results;
            auto &raw_publish_cfg = (*iter)->publish_cfg->raw_publish_cfg;
            for (int i = 0; i < raw_publish_cfg.size(); i++)
            {
                if (raw_publish_cfg[i].need_publish)
                {
                    auto data = std::make_shared<Publish_data>();
                    data->topic_base = raw_publish_cfg[i].topic_name;
                    data->data = module_output[raw_publish_cfg[i].output_result_name];
                    data->raw_publish = 1;
                    data->module_name = (*iter)->module_name;
                    publish_results.push_back(data);
                }
            }
            if ((*iter)->no_filter==false && filter_module_names_set.find((*iter)->module_name) != filter_module_names_set.end())
            {
                std::map<std::string, std::shared_ptr<InputOutput>> module_filted;
#ifdef __WITH_TRY_CATCH__

                try
                {
#endif
                    (*iter)->time_counter.start_filter(channel_name,get_time());
                    (*iter)->module->filter(channel_name, module_output, module_filted);
                    (*iter)->time_counter.end_filter(channel_name,get_time());
#ifdef __WITH_TRY_CATCH__
                }
                catch (Alg_Module_Exception &e)
                {
                    e.set_trace(e.st);
                    throw e;
                }
                catch (std::runtime_error &e)
                {
                    auto st= get_current_stacktrace_from_exception();
                    Alg_Module_Exception except_data(e.what(), (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::filter, channel_name);
                    except_data.set_trace(st);
                    throw except_data;
                }
                catch (cv::Exception &e)
                {              
                    auto st= get_current_stacktrace_from_exception();
                    Alg_Module_Exception except_data(e.msg+"\n"+e.func+"\n"+e.file+"\n"+e.err, (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::filter, channel_name);
                    except_data.set_trace(st);
                    throw except_data;
                }
                catch (...)
                {
                    auto st= get_current_stacktrace_from_exception();
                    Alg_Module_Exception except_data("unknown runtime error", (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::filter, channel_name);
                    except_data.set_trace(st);
                    throw except_data;
                }
#endif                
                if (module_filted.size() > 0)
                {
#ifdef __WITH_TRY_CATCH__
                    try
                    {
#endif
                        (*iter)->time_counter.start_display(channel_name,get_time());
                        (*iter)->module->display(channel_name, module_input, module_filted);
                        (*iter)->time_counter.end_display(channel_name,get_time());
#ifdef __WITH_TRY_CATCH__
                    }
                    catch (Alg_Module_Exception &e)
                    {
                        e.set_trace(e.st);
                        throw e;
                    }
                    catch (std::runtime_error &e)
                    {
                        auto st= get_current_stacktrace_from_exception();
                        Alg_Module_Exception except_data(e.what(), (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::display, channel_name);
                        except_data.set_trace(st);
                        throw except_data;
                    }
                    catch (cv::Exception &e)
                    {                
                        auto st= get_current_stacktrace_from_exception();
                        Alg_Module_Exception except_data(e.msg+"\n"+e.func+"\n"+e.file+"\n"+e.err, (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::display, channel_name);
                        except_data.set_trace(st);
                        throw except_data;
                    }
                    catch (...)
                    {
                        auto st= get_current_stacktrace_from_exception();
                        Alg_Module_Exception except_data("unknown runtime error", (*iter)->module->get_module_name(), Alg_Module_Exception::Stage::display, channel_name);
                        except_data.set_trace(st);
                        throw except_data;
                    }
#endif
                    //                                                     debug_show(channel_name,module_input,module_filted);
                    for (auto iter1 = module_filted.begin(); iter1 != module_filted.end(); iter1++)
                    {
                        auto data = std::make_shared<Publish_data>();
                        data->topic_base = "";
                        data->data = iter1->second;
                        data->module_name = (*iter)->module_name;
                        publish_results.push_back(data);
                    }

                    auto &filter_publish_cfg = (*iter)->publish_cfg->filter_publish_cfg;

                    for (int i = 0; i < filter_publish_cfg.size(); i++)
                    {
                        if (filter_publish_cfg[i].need_publish)
                        {
                            auto data = std::make_shared<Publish_data>();
                            data->topic_base = filter_publish_cfg[i].topic_name;
                            data->data = module_filted[filter_publish_cfg[i].output_result_name];
                            data->module_name = (*iter)->module_name;
                            publish_results.push_back(data);
                        }
                    }
                }
            }
            publish_results_all.insert(publish_results_all.end(), publish_results.begin(), publish_results.end());

//            return publish_results; }));
        }
    }
    /*
    if (this->debug)
    {
        std::cout << "wait for filter and display tasks" << std::endl;
        while (true)
        {
            int finish_cnt=0;
            for (int i = 0; i < futures.size(); i++)
            {
                if (futures[i].wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
                {
                    std::cout << "\t task " << futures_for_node[i] << " is running" << std::endl;
                }
                else{
                    finish_cnt+=1;
                }
            }
            if(finish_cnt>=futures.size()){
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    for (int i = 0; i < futures.size(); i++)
    {
        auto res = futures[i].get();
        publish_results_all.insert(publish_results_all.end(), res.begin(), res.end());
    }*/

    return publish_results_all;
};

void Alg_Engine::get_channel_status(std::string channel_name, std::vector<std::string> &out_channel_names, std::vector<uint8_t> &status, std::vector<std::string> &string)
{
    if (channel_name == "")
    {
        out_channel_names.resize(0);
        status.resize(0);
        string.resize(0);
        for (auto iter = this->channel_status.begin(); iter != this->channel_status.end(); iter++)
        {
            out_channel_names.push_back(iter->first);
            if (iter->second.status == ChannelStatus::Status::running)
            {
                status.push_back(0);
                string.push_back("");
            }
            else if (iter->second.status == ChannelStatus::Status::runtime_error)
            {
                status.push_back(1);
                string.push_back("channel runtime error");
            }
            else if (iter->second.status == ChannelStatus::Status::no_frame)
            {
                status.push_back(2);
                string.push_back("no channel data frame");
            }
            else if (iter->second.status == ChannelStatus::Status::no_config)
            {
                status.push_back(3);
                string.push_back("not set channel cfg");
            }
        }
    }
    else
    {
        out_channel_names.resize(1);
        status.resize(1);
        string.resize(1);
        out_channel_names[0] = channel_name;
        if (this->channel_status.find(channel_name) == this->channel_status.end())
        {
            status[0] = 2;
            string[0] = "no channel data frame";
        }
        else
        {
            if (this->channel_status[channel_name].status == ChannelStatus::Status::running)
            {
                status[0] = 0;
                string[0] = "";
            }
            else if (this->channel_status[channel_name].status == ChannelStatus::Status::runtime_error)
            {
                status[0] = 1;
                string[0] = "channel runtime error";
            }
            else if (this->channel_status[channel_name].status == ChannelStatus::Status::no_frame)
            {
                status[0] = 2;
                string[0] = "no channel data frame";
            }
            else if (this->channel_status[channel_name].status == ChannelStatus::Status::no_config)
            {
                status[0] = 3;
                string[0] = "not set channel cfg";
            }
        }
    }
};
std::vector<std::string> Alg_Engine::get_module_names(){
    std::shared_lock<std::shared_mutex> lock(module_edit_mutex);;
    this->to_map();
    std::vector<std::string> result;
    for(auto iter=nodes_map.begin();iter!=nodes_map.end();iter++){
        result.push_back(iter->first);
    }
    return result;

};


void Alg_Engine::enable_modules(std::vector<std::string> module_names, std::string channel_name)
{
    std::shared_lock<std::shared_mutex> lock(module_edit_mutex);;
    this->to_map();
    for (int i = 0; i < module_names.size(); i++)
    {
        if (this->nodes_map.find(module_names[i]) != this->nodes_map.end())
        {
            if (channel_name == "")
            {
                this->nodes_map[module_names[i]]->enable = true;
                this->nodes_map[module_names[i]]->channel_enable.clear();
            }
            else
            {
                this->nodes_map[module_names[i]]->channel_enable[channel_name] = true;
            }
            if (this->debug)
                std::cout << "enable module " << module_names[i] << " channel " << channel_name << std::endl;
        }
    }
};

void Alg_Engine::disable_modules(std::vector<std::string> module_names, std::string channel_name)
{
    std::shared_lock<std::shared_mutex> lock(module_edit_mutex);;
    this->to_map();
    //    this->to_map();
    for (int i = 0; i < module_names.size(); i++)
    {
        if (this->nodes_map.find(module_names[i]) != this->nodes_map.end())
        {
            if (channel_name == "")
            {
                this->nodes_map[module_names[i]]->enable = false;
                this->nodes_map[module_names[i]]->channel_enable.clear();
            }
            else
            {
                this->nodes_map[module_names[i]]->channel_enable[channel_name] = false;
            }
            if (this->debug)
                std::cout << "disable module " << module_names[i] << " channel " << channel_name << std::endl;
        }
    }
};

void Alg_Engine::get_enable_modules(std::string channel_name, std::vector<std::string> &module_names, std::vector<bool> &module_state)
{
    module_names.clear();
    module_state.clear();
    std::shared_lock<std::shared_mutex> lock(module_edit_mutex);;
    for (int i = 0; i < nodes.size(); i++)
    {
        bool enable = nodes[i]->enable;
        if (nodes[i]->channel_enable.find(channel_name) != nodes[i]->channel_enable.end())
        {
            enable = nodes[i]->channel_enable[channel_name];
        }
        module_names.push_back(nodes[i]->module_name);
        module_state.push_back(enable);
    }
};

bool Alg_Engine::check_module_enable(std::string channel_name,std::shared_ptr<Alg_Node> module){
    if(module->enable){
        if(module->channel_enable.find(channel_name) != module->channel_enable.end())
            return module->channel_enable[channel_name];
        return true;
    }
    else{
        if(module->channel_enable.find(channel_name) != module->channel_enable.end())
            return module->channel_enable[channel_name];
        return false;
    }
    
};
void Alg_Engine::set_module_enable(std::string channel_name,std::shared_ptr<Alg_Node> module,bool enable){
    module->channel_enable[channel_name]=enable;
};

void Alg_Engine::print_module_time_summary(){
    for (int i = 0; i < nodes.size(); i++)
    {
        std::cout<<nodes[i]->time_counter.to_string()<<std::endl;
    }
};
