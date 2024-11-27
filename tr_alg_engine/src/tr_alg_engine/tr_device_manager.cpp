
#include"tr_alg_engine/tr_device_manager.h"
#include <iostream>
using std::placeholders::_1;
using std::placeholders::_2;


    bool Device_info::check_memory(float mem_reques){
        return avaliable_mem_mbytes-protect_mem-mem_reques>0;
    };

    bool Device_info::check_util(float util_request){
        if(100-util<10)
            return false;
        return 100-util-util_request>0;
    };

    std::string Device_info::str(){
        return "tpu kernel id: "+Num2string<int>(this->device_id)+"\t util: "+Num2string<int>(this->util)+"\t mem: "
            +Num2string<int>(this->total_mem_mbytes)+"Mb\t ava: "+Num2string<int>(this->avaliable_mem_mbytes)+"Mb";
    };
    bool Host_info::check_memory(float mem_reques){
        return avaliable_mem_mbytes-protect_mem-mem_reques>0;
    };

    bool Host_info::check_util(float util_request){
        return 100-cpu_current_average-util_request/cpu_max>0;
    };
    std::string Host_info::str(){
        return "cpu num: "+Num2string<int>(this->cpu_max)+"\t util: "+Num2string<float>(this->cpu_current_average)+"\t mem: "
            +Num2string<int>(this->total_mem_mbytes)+"Mb\t ava: "+Num2string<int>(this->avaliable_mem_mbytes)+"Mb";
    };


    Task::Task(){


    };
    Task::Task(int memory_mbytes,int tpu_memory_mbytes,int cpu_load,int tpu_load){
        this->memory_mbytes=memory_mbytes;
        this->tpu_memory_mbytes=tpu_memory_mbytes;
        this->cpu_load=cpu_load;
        this->tpu_load=tpu_load;
    };

    Task::Task(const Task & other){
        this->count_down=other.count_down;
        this->max_count=other.max_count;
        this->memory_mbytes=other.memory_mbytes;
        this->tpu_memory_mbytes=other.tpu_memory_mbytes;
        this->tpu_load=other.tpu_load;
        this->cpu_load=other.cpu_load;
        this->device_id=other.device_id;
    };
    void Task::tick(){
        this->count_down+=1;
    };
    bool Task::timeout(){
        return this->count_down>this->max_count;
    };
    void Task::operator=(const Task & other){
        this->count_down=other.count_down;
        this->max_count=other.max_count;
        this->memory_mbytes=other.memory_mbytes;
        this->tpu_memory_mbytes=other.tpu_memory_mbytes;
        this->tpu_load=other.tpu_load;
        this->cpu_load=other.cpu_load;
        this->device_id=other.device_id;
    };


    void Task_prepare::push_task(Task& task){
        this->tasks.push_back(task);
    };
    void Task_prepare::tick(){
        std::vector<Task>  tasks_new;
        for(auto iter=tasks.begin();iter!=tasks.end();iter++){
            iter->tick();
            if(iter->timeout()==false){
                tasks_new.push_back(*iter);
            }
        }
        if(tasks_new.size()!=tasks.size())
            tasks=tasks_new;
    };
    std::map<int,float> Task_prepare::check_new_task_with_util_res(Task& new_task,std::vector<Device_info> devices_info,Host_info host_info,std::vector<unsigned int> device_idx,bool force_get){
        std::vector<Device_info> avaliable_devices;
        for(int j=0;j<device_idx.size();j++){
            int i=device_idx[j];
            if(devices_info[i].check_memory(new_task.tpu_memory_mbytes)){
                if(force_get||devices_info[i].check_util(new_task.tpu_load))
                    avaliable_devices.push_back(devices_info[i]);
            }
        }
        std::map<int,float> res;
        if(avaliable_devices.size()<=0){

            return res;
        }

        std::sort(avaliable_devices.begin(),avaliable_devices.end(),[](const Device_info& a,const Device_info& b){
            float score_a=(float)a.avaliable_mem_mbytes/a.total_mem_mbytes+(float)a.util/100.0*0.2;
            float score_b=(float)b.avaliable_mem_mbytes/b.total_mem_mbytes+(float)b.util/100.0*0.2;
            return score_a<score_b;
        });
        for(auto iter=avaliable_devices.begin();iter!=avaliable_devices.end();iter++){
            float score=(float)iter->avaliable_mem_mbytes/iter->total_mem_mbytes+(100-(float)iter->util)/100.0*2;
            res.insert(std::make_pair(iter->device_id,score));
        }

        return res;

    }

    int Task_prepare::check_new_task(Task& new_task,std::vector<Device_info> devices_info,Host_info host_info,std::vector<unsigned int> device_idx,bool force_get){
        this->add_prepare_source(devices_info,host_info);        
        if(host_info.check_memory(new_task.memory_mbytes)==false){

            return -1;
        }
        if(!force_get&&host_info.check_util(new_task.cpu_load)==false){
            return -2;

        }
        std::vector<Device_info> avaliable_devices;
        for(int j=0;j<device_idx.size();j++){
            int i=device_idx[j];
            if(devices_info[i].check_memory(new_task.tpu_memory_mbytes)){
                if(force_get||devices_info[i].check_util(new_task.tpu_load))
                    avaliable_devices.push_back(devices_info[i]);
            }
        }
        if(avaliable_devices.size()<=0){

            return -3;
        }

        std::sort(avaliable_devices.begin(),avaliable_devices.end(),[](const Device_info& a,const Device_info& b){
            float score_a=(float)a.avaliable_mem_mbytes/a.total_mem_mbytes+(100-(float)a.util)/100.0*2;
            float score_b=(float)b.avaliable_mem_mbytes/b.total_mem_mbytes+(100-(float)b.util)/100.0*2;
            return score_a>score_b;
        });
        return avaliable_devices[0].device_id;

    };

    void Task_prepare::add_prepare_source(std::vector<Device_info>& devices_info,Host_info& host_info){
        for(auto iter=tasks.begin();iter!=tasks.end();iter++){
            for(int i=0;i<devices_info.size();i++){
                if(iter->device_id>=0 && devices_info[i].device_id==iter->device_id){
                    devices_info[i].avaliable_mem_mbytes-=iter->tpu_memory_mbytes;
                    devices_info[i].util+=iter->tpu_load;
                }
                host_info.avaliable_mem_mbytes-=iter->memory_mbytes;
                host_info.cpu_current_average+=iter->cpu_load/host_info.cpu_max;
            }
        }
    };


Alg_Node_Device_Manager::Alg_Node_Device_Manager(std::string dev_platform_name) : Alg_Node_Base("Device_manager_node",dev_platform_name){
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&Alg_Node_Device_Manager::update, this));

    device_usage_publisher = this->create_publisher<tr_alg_interfaces::msg::DeviceUsage>("/device_usage_"+dev_platform_name, 10);
    this->get_device_Server=this->create_service<tr_alg_interfaces::srv::GetAvaliableDevice>("/service/device_"+dev_platform_name+"/get_avaliable_device",
        std::bind(&Alg_Node_Device_Manager::get_device,this,_1,_2));
//    );
    moniter=get_device_stat();
}

void Alg_Node_Device_Manager::reset_devices_handle(){
    int device_count=get_device_count();
    {

        if(device_count>this->devices_handle.size()){
            std::vector<std::shared_ptr<Device_Handle>> new_devices_handle;
            new_devices_handle.resize(device_count);

            std::vector<unsigned int > new_devices_card_idx;
            std::vector<Device_info> new_devices_info;
            new_devices_card_idx.resize(device_count);
            new_devices_info.resize(device_count);
            for(int i=0;i<new_devices_card_idx.size();i++){
                new_devices_card_idx[i]=1000;
            }

            std::set<int> in_used_device_ids;
            for(int i=0;i<this->devices_card_idx.size();i++){
                in_used_device_ids.insert(this->devices_card_idx[i]);
                new_devices_handle[this->devices_card_idx[i]]=this->devices_handle[i];
                new_devices_card_idx[this->devices_card_idx[i]]=this->devices_card_idx[i];
                new_devices_info[this->devices_card_idx[i]]=this->devices_info[i];
            }
            for(int i=0;i<device_count;i++){
                if(in_used_device_ids.find(i)==in_used_device_ids.end()){
                    new_devices_handle[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
                    try{
                    Device_dev_stat stat= moniter->get_device_stat(new_devices_handle[i]);
                    new_devices_info[i].device_id=new_devices_handle[i]->get_device_id();
                    new_devices_info[i].total_mem_mbytes=stat.mem_total;
                    new_devices_info[i].avaliable_mem_mbytes=stat.mem_total-stat.mem_used;
                    new_devices_info[i].util=stat.tpu_util;

                    }                    
                    catch(...){
                        
                    }
                }
            }
            std::vector<std::shared_ptr<Device_Handle>> res_devices_handle;
            std::vector<unsigned int > res_devices_card_idx;
            std::vector<Device_info> res_devices_info;
            for(int i=0;i<new_devices_card_idx.size();i++){
                if(new_devices_card_idx[i]>=1000){

                }
                else{
                    res_devices_handle.push_back(new_devices_handle[i]);
                    res_devices_card_idx.push_back(new_devices_card_idx[i]);
                    res_devices_info.push_back(new_devices_info[i]);
                }
            }
            std::unique_lock lock(this->mutex);
            this->devices_handle=res_devices_handle;
            this->devices_card_idx=res_devices_card_idx;
            this->devices_info=res_devices_info;


        }

    }

};

void Alg_Node_Device_Manager::init_devices_handle(){
    int device_count=get_device_count();
    {
    
        if(device_count>0){
            std::unique_lock lock(this->mutex);
            this->devices_handle.resize(device_count);
            this->devices_card_idx.resize(device_count);
            int idx=0;
            for(int i=0;i<device_count;i++){
                this->devices_handle[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
                if(this->devices_handle[i]!=nullptr){
                    this->devices_card_idx[i]=this->devices_handle[i]->get_card_id();
                    idx+=1;
                }

            }
            this->devices_handle.resize(idx);
            this->devices_card_idx.resize(idx);
            RCLCPP_INFO(this->get_logger(), "INFO:\t %d device id inited",idx);
            for(int i=0;i<devices_card_idx.size();i++){
                if(device_card_map.find(devices_card_idx[i])==device_card_map.end()){
                    device_card_map[devices_card_idx[i]]=std::vector<unsigned int>();
                }
                int tpu_id=devices_handle[i]->get_device_id();
                device_card_map[devices_card_idx[i]].push_back(tpu_id);
            }

        }
    }

};

void Alg_Node_Device_Manager::init_devices_info(){
    if(this->devices_handle.size()>0 && this->devices_info.size()<=0){
        std::unique_lock lock(this->mutex);
        this->devices_info.resize(this->devices_handle.size());


        for(int i=0;i<this->devices_info.size();i++){
            Device_dev_stat device_stats=moniter->get_device_stat(devices_handle[i]);

            int tpu_id=devices_handle[i]->get_device_id();
            this->devices_info[i].device_id=tpu_id;
            this->devices_info[i].total_mem_mbytes=device_stats.mem_total;
            this->devices_info[i].avaliable_mem_mbytes=device_stats.mem_total-device_stats.mem_used;
            this->devices_info[i].util=device_stats.tpu_util;
        }
    }
};
void Alg_Node_Device_Manager::init_host_info(){

    moniter->init();

};

void Alg_Node_Device_Manager::update_devices_info(){
    if(this->devices_handle.size()>0){

        std::unique_lock lock(this->mutex);

        for(int i=0;i<this->devices_info.size();i++){
            Device_dev_stat device_stats=moniter->get_device_stat(devices_handle[i]);

            this->devices_info[i].avaliable_mem_mbytes=device_stats.mem_total-device_stats.mem_used;
            this->devices_info[i].util=device_stats.tpu_util;

        }    
    }
};
void Alg_Node_Device_Manager::update_host_info(){
    Host_dev_stat host_stat=moniter->get_host_stat();    
    std::unique_lock lock(this->mutex);
    this->host_info.total_mem_mbytes=host_stat.mem_total;
    this->host_info.avaliable_mem_mbytes=host_stat.mem_total-host_stat.mem_used;
    this->host_info.cpu_current_average=host_stat.cpu_util;

};
void Alg_Node_Device_Manager::update_tasks(){
    this->tasks.tick();
};
 

void Alg_Node_Device_Manager::update(){
    if(this->is_first){
        this->init_host_info();
        this->init_devices_handle();
        this->init_devices_info();
        this->update_host_info();
        this->is_first=false;


    }
    else{
        this->reset_devices_handle();
        this->update_devices_info();
        this->update_host_info();
    }

    if(this->devices_handle.size()<=0){
        this->is_first=true;
        return;
    }

    tr_alg_interfaces::msg::DeviceUsage message;
    message.memory_mbyte=(host_info.total_mem_mbytes-host_info.avaliable_mem_mbytes);
    message.memory=message.memory_mbyte/host_info.total_mem_mbytes;
    message.cpu_use=host_info.cpu_current_average;
    for(int i=0;i<this->devices_info.size();i++){
        float mbyte=devices_info[i].total_mem_mbytes-devices_info[i].avaliable_mem_mbytes;
        message.tpu_id.push_back(devices_info[i].device_id);
        message.tpu_memory.push_back(mbyte/devices_info[i].total_mem_mbytes);
        message.tpu_memory_mbyte.push_back(mbyte);
        message.tpu_use.push_back(devices_info[i].util);
    }
    this->device_usage_publisher->publish(message);
    this->update_tasks();
};

void Alg_Node_Device_Manager::get_device(const tr_alg_interfaces::srv::GetAvaliableDevice::Request::SharedPtr request,
    const tr_alg_interfaces::srv::GetAvaliableDevice::Response::SharedPtr response){
    RCLCPP_INFO(this->get_logger(), "get device request");

    Task new_task(request->memory,request->gpu_memory,request->cpu_load,request->gpu_load);
//    response.device_id.resize(device_card_map.size());
    bool force_get=request->force_get>0;
    int i=0;
    for(auto iter=this->device_card_map.begin();iter!=device_card_map.end();iter++,i++){
        int device_id=-5;
        {
            std::unique_lock lock(this->mutex);
            device_id=this->tasks.check_new_task(new_task,this->devices_info,this->host_info,iter->second,force_get);
            response->device_id.push_back(device_id);
        }
        if(device_id>=0){
            new_task.device_id=device_id;
            auto res=this->tasks.check_new_task_with_util_res(new_task,this->devices_info,this->host_info,iter->second,force_get);
            for(auto iter=res.begin();iter!=res.end();iter++){
                RCLCPP_INFO(this->get_logger(), "INFO:\t device %d score %f",iter->first,iter->second);
            }
            this->tasks.push_task(new_task);
            RCLCPP_INFO(this->get_logger(), "INFO:\t provide device %d",device_id);

        }
        else{
            if(device_id==-1||device_id==-2){
                RCLCPP_INFO(this->get_logger(), "INFO:\t host resource not enough %s",this->host_info.str().c_str());
            }
            else if(device_id==-3){
                for(int i=0;i<this->devices_info.size();i++){
                    RCLCPP_INFO(this->get_logger(), "INFO:\t device resource not enough %s",this->devices_info[i].str().c_str());
                }
            }

            response->message="reject request because of the high resource occupancy at current";
        }
        std::cout<<"prob device id:"<<device_id<<std::endl;
    }
    if(response->device_id.size()!=this->device_card_map.size()){
        response->device_id.clear();
        response->message="some tpu card is error";
    }
//    int device_id=this->tasks.check_new_task(new_task,this->devices_info,this->host_info,);
//    response->device_id=device_id;
};
