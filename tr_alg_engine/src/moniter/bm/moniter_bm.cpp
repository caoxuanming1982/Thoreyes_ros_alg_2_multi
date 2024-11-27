#include "moniter/bm/moniter_bm.h"

#include<bmruntime.h>
#include<bmruntime_cpp.h>
#include<bmcv_api.h>

#include "network_engine/bm/device_handle_bm.h"
Device_dev_stat Moniter_bm::get_device_stat(std::shared_ptr<Device_Handle> device)
{

    std::shared_ptr<Device_Handle_bm> handle=std::dynamic_pointer_cast<Device_Handle_bm>(device);
    Device_dev_stat res;
    if(handle!=nullptr){
        bm_dev_stat_t stat;
        bm_get_stat(handle->handle,&stat);
        res.mem_total=stat.mem_total;
        res.mem_used=stat.mem_used;
        res.tpu_util=stat.tpu_util;
    }
    else{

        throw std::runtime_error("use bm operate in not sophon device");
    }

    //        std::cout<<idx<<"\t"<<res.mem_total<<"\t"<<res.mem_used<<"\t"<<res.tpu_util<<std::endl;
    return res;
};
std::shared_ptr<Moniter> get_device_stat(){
    return std::make_shared<Moniter_bm>();
};


int get_device_count(){
    int count=0;
    bm_dev_getcount(&count);
    return count;
};
