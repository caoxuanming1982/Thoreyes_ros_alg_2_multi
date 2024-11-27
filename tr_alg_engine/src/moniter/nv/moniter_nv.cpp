#include "moniter/nv/moniter_nv.h"
#include <cuda_runtime.h>
#include <nvml.h>

Device_dev_stat Moniter_nv::get_device_stat(std::shared_ptr<Device_Handle> device)
{
    nvmlInit();
    Device_dev_stat res;
    cudaDeviceProp deviceProp;
    int idx = device->get_device_id();
    cudaGetDeviceProperties(&deviceProp, idx);
    size_t freeMemory, totalMemory;
    cudaSetDevice(idx);
    cudaMemGetInfo(&freeMemory, &totalMemory);
    nvmlDevice_t device1;
    nvmlDeviceGetHandleByIndex(idx, &device1);
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(device1, &util);
    res.mem_total = totalMemory / 1024 / 1024;
    res.mem_used = totalMemory / 1024 / 1024 - freeMemory / 1024 / 1024;
    res.tpu_util = util.gpu;
    //        std::cout<<idx<<"\t"<<res.mem_total<<"\t"<<res.mem_used<<"\t"<<res.tpu_util<<std::endl;
    return res;
};

std::shared_ptr<Moniter> get_device_stat(){
    return std::make_shared<Moniter_nv>();
}

int get_device_count(){
    int count=0;
    cudaGetDeviceCount(&count);
    return count;
};
