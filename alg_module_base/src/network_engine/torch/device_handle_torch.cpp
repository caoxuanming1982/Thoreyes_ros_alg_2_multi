#include "network_engine/torch/device_handle_torch.h"


int Device_Handle_torch::get_device_id(){
    
    return handle.index();
};
int Device_Handle_torch::get_card_id(){
    return handle.index();
};
Device_Handle* get_device_handle(int idx){
    torch::Device dev(c10::DeviceType::CUDA,idx);
    return new Device_Handle_torch(dev);
};
void free_device_handle(Device_Handle* handle){
    delete handle;
};