#include "network_engine/network_parallel.h"

Unit_counter::Unit_counter(){
    
};

void Unit_counter::tick_start_time(long start_time)
{
    auto lock=std::unique_lock(mutex);
    if (last_start_time != 0)
    {
        time_intervals.push_back(start_time - last_start_time);
    }

    last_start_time = start_time;
    if (time_intervals.size() > this->max_counter_cnt)
    {
        time_intervals.erase(time_intervals.begin());
    }
};
void Unit_counter::tick_busy_time(long busy_time)
{
    auto lock=std::unique_lock(mutex);
    busy_time_intervals.push_back(busy_time);
    if (busy_time_intervals.size() > this->max_counter_cnt)
    {
        busy_time_intervals.erase(busy_time_intervals.begin());
    }
};
void Unit_counter::reset()
{
    auto lock=std::unique_lock(mutex);
    long last_start_time = 0;
    time_intervals.resize(0);
    busy_time_intervals.resize(0);
};
float Unit_counter::get_util()
{
    auto lock=std::shared_lock(mutex);
    int cnt = time_intervals.size() - 1;
    if (cnt > busy_time_intervals.size())
    {
        cnt = busy_time_intervals.size();
    }
    if (cnt <= 5)
        return -1;

    float time_total=0;// = time_intervals[cnt-1]-time_intervals[0];
    float time_use = 0;
    for (int i = 0; i < cnt; i++)
    {
        time_total += time_intervals[i];
        time_use += busy_time_intervals[i];
    }
    return time_use / time_total;
};

Network_instance::Network_instance(std::shared_ptr<Device_Handle> handle,std::string file_name,std::string model_name,std::vector<Shape_t>& input_shapes){
    this->handle = handle;
    this->file_name = file_name;
    this->model_name = model_name;
    this->cache_inputs_shapes=input_shapes;


};

std::shared_ptr<Device_Handle> Network_instance::get_handle()
{
    return handle;
};
bool Network_instance::reload_model(std::string file_name, std::string model_name)
{
    auto lock = std::unique_lock(mutex);
    free_model();
    this->file_name = file_name;
    this->model_name = model_name;
    return load_model();
}
bool Network_instance::load_model()
{
    if (access(this->file_name.c_str(), F_OK) != 0)
        return false;

    auto lock = std::unique_lock(mutex);
    free_model();
    instance=get_network_kernel(this->handle,this->file_name,this->model_name,cache_inputs_shapes);

    return true;
};
bool Network_instance::free_model()
{
    free_network_kernel(instance);
    instance = nullptr;
    return true;
};
Network_instance::~Network_instance(){
    {
        auto lock = std::unique_lock(mutex);
        free_model();
    }
};

int Network_instance::forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs){
    auto lock = std::shared_lock(mutex);
    long start_timestamp = get_time();
    this->util_counter.tick_start_time(start_timestamp);
    int res = this->instance->forward(inputs, outputs);
    long end_timestamp = get_time();
    this->util_counter.tick_busy_time(end_timestamp - start_timestamp);
    return res;

};

int Network_instance::forward(std::vector<cv::Mat>& inputs,std::vector<Output>& outputs){
    auto lock = std::shared_lock(mutex);
    long start_timestamp = get_time();
    this->util_counter.tick_start_time(start_timestamp);
    std::vector<std::shared_ptr<QyImage>> inputs_;
    for(int i=0;i<inputs.size();i++){
        inputs_.push_back(from_mat(inputs[i],handle));
    }
    int res = this->instance->forward(inputs_, outputs);
    long end_timestamp = get_time();
    this->util_counter.tick_busy_time(end_timestamp - start_timestamp);
    return res;

};
void Network_instance::set_input_scale_offset(std::vector<cv::Scalar> scale,std::vector<cv::Scalar> offset){
    this->instance->input_scale=scale;
    this->instance->input_offset=offset;
};



Network_instance *Network_instance::copy(std::shared_ptr<Device_Handle> handle)
{
    Network_instance *res = new Network_instance(handle, this->file_name, this->model_name,cache_inputs_shapes);
    res->load_model();
    return res;
};

float Network_instance::get_util(){
    return this->util_counter.get_util();
};


std::vector<Shape_t> Network_instance::get_input_shapes(){
    if(this->instance==nullptr)
        return std::vector<Shape_t>();
    return this->instance->get_input_shapes();
};


void Network_instance::set_input_shapes(std::vector<Shape_t>& shapes)
{
    this->cache_inputs_shapes=shapes;
    this->instance->cache_inputs_shapes=shapes;
    
}

Network_parallel::Network_parallel(){

};
void Network_parallel::set_device_ids(std::vector<int> device_id)
{
    for (int i = 0; i < device_id.size(); i++)
    {
        Device_Handle* hd= get_device_handle(device_id[i]);
        this->device_handles.insert(std::make_pair(device_id[i], hd));

    }
};

void Network_parallel::set_device_handles(std::map<int,std::shared_ptr<Device_Handle>> device_handles){

    this->device_handles=device_handles;
};

void Network_parallel::load_model(std::string file_name, std::string model_name)
{
    file_name=convert_model_path(file_name);
    this->file_name = file_name;
    this->model_name = model_name;
    for (int i = 0; i < instances.size(); i++)
    {
        instances[i]->reload_model(file_name, model_name);
    }
};

bool Network_parallel::add_inference_instance(int device_id)
{
    if(this->instances.size()>=this->max_instance_cnt)
        return false;
    std::shared_ptr<Network_instance> new_instance;
    {

        auto lock=std::unique_lock(mutex);
        if (this->device_handles.find(device_id) == this->device_handles.end())
        {
            return false;
        }
        new_instance =std::make_shared<Network_instance>(this->device_handles[device_id], this->file_name, this->model_name,this->cache_inputs_shapes);
    }
    if (new_instance != nullptr)
    {
        bool ret = new_instance->load_model();
        if (ret){
            for(int i=0;i<instances.size();i++){
                instances[i]->util_counter.reset();
            }
            new_instance->set_input_scale_offset(input_scale,input_offset);

            instances.push_back(new_instance);
            processor_queue.push(new_instance);
        }
        else{
            new_instance.reset();
        }
        return ret;
    }
    return false;
};
bool Network_parallel::remove_last_inference_instance()
{
    auto lock=std::unique_lock(mutex);
    if (instances.size() <= 1)
    {
        return false;
    }
    std::map<int,float> max_util_per_card;
    std::map<int,int> max_idx_per_card;
    std::map<int,int> n_ins_per_card;

    for(int i=0;i<instances.size();i++){
        float ins_util=this->instances[i]->get_util();
        unsigned int temp=0;
        temp=instances[i]->handle->get_device_id();
        if(max_util_per_card.find(temp)==max_util_per_card.end()){
            max_util_per_card[temp]=ins_util;
            max_idx_per_card[temp]=i;
            n_ins_per_card[temp]=1;
        }
        else{
            if(max_util_per_card[temp]<ins_util){
                max_util_per_card[temp]=ins_util;
                max_idx_per_card[temp]=i;
            }
            n_ins_per_card[temp]+=1;

        }
    }
    std::vector<float> max_utils;
    std::vector<int> max_idxes;
    for(auto iter=n_ins_per_card.begin();iter!=n_ins_per_card.end();iter++){
        if(iter->second>1){
            max_utils.push_back(max_util_per_card[iter->first]);
            max_idxes.push_back(max_idx_per_card[iter->first]);
        }
    }
    float max_util=-1;
    int max_idx=-1;
    for(int i=0;i<max_utils.size();i++){
        if(max_util<max_utils[i]){
            max_util=max_utils[i];
            max_idx=max_idxes[i];
        }
    }

    if(max_idx>=0){
        std::shared_ptr<Network_instance> last_instance = instances[max_idx];
        instances[max_idx]=nullptr;
        instances.erase(instances.begin()+max_idx);
        std::queue<std::shared_ptr<Network_instance>> temp;
        while (processor_queue.size() > 0)
        {
            if (processor_queue.front() != last_instance)
            {
                temp.push(processor_queue.front());
            }
            processor_queue.pop();
        }
        processor_queue = temp;
        return true;

    }
    else{
        return false;
    }


};

std::shared_ptr<Network_instance> Network_parallel::get_instance(int card_id){
    int input_card_id=card_id;
    std::shared_ptr<Network_instance> instance ;
    {

        auto lock=std::unique_lock(mutex);
        if (instances.size() < 1)
        {
            return instance;
        }
        while (true){
            if(processor_queue.size() <= 0){
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.lock();
                
            }
            else{
                instance = processor_queue.front();
                processor_queue.pop();
                unsigned int temp=0;
                temp=instance->get_handle()->get_card_id();
                int net_card_id=temp;
//                std::cout<<input_card_id<<" "<<net_card_id<<std::endl;
                if(input_card_id>=0 && input_card_id!=net_card_id){

                    processor_queue.push(instance);
                    instance=nullptr;
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    lock.lock();

                }
                else{
                    break;
                }
            }
        }

    }

    return instance;
}

int Network_parallel::forward(std::vector<std::shared_ptr<QyImage>>& inputs,std::vector<Output>& outputs){
    int input_card_id=-1;
    if (inputs.size()>0){
        input_card_id=inputs[0]->get_handle()->get_card_id();
    }
    std::shared_ptr<Network_instance> instance =get_instance(input_card_id);
    if (instance == nullptr)
        return -1;

    int current_device_id = instance->get_handle()->get_device_id();
    instance->forward(inputs, outputs);
    {
        auto lock=std::unique_lock(mutex);
        processor_queue.push(instance);
    }

    return 0;

}


int Network_parallel::forward(std::vector<cv::Mat>& inputs,std::vector<Output>& outputs){
    int input_card_id=-1;
    std::shared_ptr<Network_instance> instance =get_instance(input_card_id);
    if (instance == nullptr)
        return -1;
    int current_device_id = instance->get_handle()->get_device_id();
    instance->forward(inputs, outputs);
    {
        auto lock=std::unique_lock(mutex);
        processor_queue.push(instance);
    }

    return 0;

};

void Network_parallel::set_input_scale_offset(std::map<int,cv::Scalar> scale,std::map<int,cv::Scalar> offset){
    std::vector<cv::Scalar> scale_in;
    std::vector<cv::Scalar> offset_in;
    int max_num_input=0;
    for(auto iter=scale.begin();iter!=scale.end();iter++){
        max_num_input=std::max(max_num_input,iter->first+1);
    }

    for(auto iter=offset.begin();iter!=offset.end();iter++){
        max_num_input=std::max(max_num_input,iter->first+1);
    }

    scale_in.resize(max_num_input,cv::Scalar(1,1,1));
    offset_in.resize(max_num_input,cv::Scalar(0,0,0));
    for(auto iter=scale.begin();iter!=scale.end();iter++){
        scale_in[iter->first]=iter->second;;   
    }
    for(auto iter=offset.begin();iter!=offset.end();iter++){
        offset_in[iter->first]=iter->second;;   
    }

    this->input_scale=scale_in;
    this->input_offset=offset_in;



    for(int i=0;i<this->instances.size();i++){
        if(this->instances[i]!=nullptr)
            this->instances[i]->set_input_scale_offset(scale_in,offset_in);
    }
};

float Network_parallel::check_util(){
    if(this->instances.size()<=0)
        return 100;
    float util=0;
    for(int i=0;i<this->instances.size();i++){
        float ins_util=this->instances[i]->get_util();
        if(ins_util<0)
            return 0;
        util+=ins_util;
    }

    util/=this->instances.size();
//    std::cout<<this->model_name<<"\t"<<util<<"\t"<<this->instances.size()<<std::endl;

    return util;
//    if(util>0.8)
  //      return 1;
    //if(util*this->instances.size()/(this->instances.size()-1)<0.7)
      //  return -1;
//    return 0;
};

std::vector<float> Network_parallel::get_util(){
    std::vector<float> res;
    for(int i=0;i<this->instances.size();i++){
        float ins_util=this->instances[i]->get_util();
        res.push_back(ins_util);
    }
    return res;

};

std::vector<Shape_t> Network_parallel::get_input_shapes(){

    if(this->instances.size()<=0){
        return std::vector<Shape_t>();
    }

    return this->instances[0]->get_input_shapes();
};


void Network_parallel::set_input_shapes(std::vector<Shape_t>& shapes)
{
    this->cache_inputs_shapes=shapes;
    for(int i=0;i<instances.size();i++){

        this->instances[i]->set_input_shapes(shapes);
    }
    
}