#include "moniter/hw/moniter_hw.h"


#include "network_engine/hw/device_handle_hw.h"
Device_dev_stat Moniter_hw::get_device_stat(std::shared_ptr<Device_Handle> device)
{

    std::shared_ptr<Device_Handle_hw> handle=std::dynamic_pointer_cast<Device_Handle_hw>(device);
    Device_dev_stat res;
    if(handle!=nullptr){
        aclrtUtilizationInfo utilization_info;
        aclrtGetDeviceUtilizationRate(handle->get_device_id(), &utilization_info);
        int32_t util=0;
        int cnt=0;
        if(utilization_info.cubeUtilization>=0){
            util+=utilization_info.cubeUtilization;
            cnt+=1;
        }
        if(utilization_info.vectorUtilization>=0){
            util+=utilization_info.vectorUtilization;
            cnt+=1;
        }
        if(utilization_info.aicpuUtilization>=0){
            util+=utilization_info.aicpuUtilization;
            cnt+=1;
        }
        if(cnt>0){
            util=util/cnt;
        }
        else{
            util=100;
        }
        res.tpu_util=util;
        size_t free,total;
        aclrtGetMemInfo(aclrtMemAttr::ACL_DDR_MEM, &free, &total);
        res.mem_total=total/1024/1024;
        res.mem_used=(total-free)/1024/1024;

    }
    else{

        throw std::runtime_error("use bm operate in not sophon device");
    }

    //        std::cout<<idx<<"\t"<<res.mem_total<<"\t"<<res.mem_used<<"\t"<<res.tpu_util<<std::endl;
    return res;
};
std::shared_ptr<Moniter> get_device_stat(){
    return std::make_shared<Moniter_hw>();
};


int get_device_count(){
    uint32_t count=0;
    aclrtGetDeviceCount(&count);
    return count;
};
