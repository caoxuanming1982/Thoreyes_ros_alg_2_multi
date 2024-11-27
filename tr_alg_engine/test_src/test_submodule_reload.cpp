
#include "alg_engine.h"
#include "moniter/moniter.h"
#include <network_engine/device_handle.h>
#include <iostream>
#include <fstream>
#include <vector>

void analysis_memory(int start,std::vector<int>& mem_info,float thres){
    int over_cnt=0;
    for(int i=0;i<mem_info.size();i++){
        if(mem_info[i]-start>=thres){
            over_cnt+=1;
        }
    }
    if(over_cnt>(mem_info.size())/2){
        std::cout<<"high confidence memory operation error"<<std::endl;
    }
    else if(over_cnt>(mem_info.size())/10){
        std::cout<<"low confidence memory operation error"<<std::endl;
    }
    else if(over_cnt>0){
        std::cout<<"lowest confidence memory operation error"<<std::endl;
    }
    else{
        std::cout<<"memory operation health"<<std::endl;
    }
    

}

int get_mem_usage(){
    std::ifstream statusFile("/proc/self/status");
    if (!statusFile.is_open()) {
        std::cerr << "Failed to open /proc/self/status" << std::endl;
        return 0;
    }

    std::string line;
    int result=0;
    while (getline(statusFile, line)) {
        if (line.find("VmRSS:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            long rss;
            iss >> key >> rss; // Read "VmRSS:" and the value in kilobytes
//            std::cout << "Current process RSS memory usage: " << rss << " KB" << std::endl;
            result=rss;
            break;
        }
    }

    statusFile.close();
    return result;
}

int main(){
    std::string lib_dir="/data_temp/thoreyes/ros/alg_module_submodules/lib/";
    std::string requirement_dir="/data_temp/thoreyes/ros/requirement/";
    std::map<std::string,std::string> channel_cfg_paths;
    channel_cfg_paths["channel_1"]="/data/thoreyes/conf/business/channels/algo_cfg_1.xml";
    channel_cfg_paths["channel_2"]="/data/thoreyes/conf/business/channels/algo_cfg_2.xml";
    channel_cfg_paths["channel_3"]="/data/thoreyes/conf/business/channels/algo_cfg_3.xml";
    channel_cfg_paths["channel_4"]="/data/thoreyes/conf/business/channels/algo_cfg_4.xml";
    channel_cfg_paths["channel_5"]="/data/thoreyes/conf/business/channels/algo_cfg_5.xml";

    int device_count=get_device_count();
    std::vector<std::shared_ptr<Device_Handle>> devices;
        devices.resize(device_count);
        for(int i=0;i<device_count;i++){
            devices[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
        }

    Alg_Engine engine;
    engine.load_module_from_libdir(lib_dir, requirement_dir)  ;  

    auto res = engine.check_same_node_name();
    if (res.size() > 0)
    {
        std::string error_string = "mult module have same name\n";
        for (int i = 0; i < res.size(); i++)
        {
            error_string += "\t" + res[i] + "\n";
        }
        std::cout <<error_string<<std::endl;
        return false;
    }
    auto res1 = engine.check_node_require();
    if (res1.size() > 0)
    {
        std::cout << "module require not meet" << std::endl;
        for (int i = 0; i < res1.size(); i++)
        {
            std::cout << "\t" << res1[i].error_msg << "\t" << res1[i].module_name << " : " << res1[i].param_name << std::endl;
        }

        return false;
    }
    auto res2 = engine.check_node_publish();
    if (res2.size() > 0)
    {
        std::cout << "module publish error" << std::endl;
        for (int i = 0; i < res2.size(); i++)
        {
            std::cout << "\t" << res2[i].error_msg << "\t" << res2[i].module_name << " : " << res2[i].param_name << std::endl;
        }
        return false;
    }

    engine.set_device_handles(devices);
    std::vector<Request_Model_instance_data> instances=engine.update_and_check_model_util(true);
    for(int i=0;i<instances.size();i++){
        instances[i].result_device_id=0;
    }
    engine.update_model_instance_num(instances);

    for(auto iter=channel_cfg_paths.begin();iter!=channel_cfg_paths.end();iter++){
        engine.set_channel_cfg(iter->first,iter->second);

    }
    std::vector<std::string> module_names= engine.get_module_names();
    int thres=10;
    std::vector<std::vector<int>> mem_usage_log;
    std::vector<int> mem_usage_start;
    int n_log=1000;
    mem_usage_log.resize(module_names.size());
    mem_usage_start.resize(module_names.size());
    for(int i=0;i<module_names.size();i++){
        mem_usage_log[i].resize(n_log);
    }
    for(int i=0;i<module_names.size();i++){
//        std::cout<<"start check module "<<module_names[i]<<std::endl;
        mem_usage_start[i]=get_mem_usage();
        for(int j=0;j<1000;j++){
            
            engine.reload_module(module_names[i]);
            std::vector<Request_Model_instance_data> instances=engine.update_and_check_model_util(false);
            for(int i=0;i<instances.size();i++){
                instances[i].result_device_id=0;
            }
            engine.update_model_instance_num(instances);

            int state;
            mem_usage_log[i][j]=get_mem_usage();
//            states.push_back(state);
        }
//        analysis_memory(start_state,states,thres);
  //      std::cout<<"end check module "<<module_names[i]<<std::endl;
    }

    for(int i=0;i<module_names.size();i++){
        std::cout<<module_names[i]+"\t\t";
        int start=mem_usage_start[i];
        for(int j=0;j<std::min(10,int(mem_usage_log[i].size()/10));j++){
            start=std::max(start,mem_usage_log[i][j]);
        }
        analysis_memory(start,mem_usage_log[i],thres);

    }

    return 1;

}