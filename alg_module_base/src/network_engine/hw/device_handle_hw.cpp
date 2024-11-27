#include "network_engine/hw/device_handle.hw.h"

int Device_Handle_hw::get_device_id(){
    
    return deviceId;
};
int Device_Handle_hw::get_card_id(){
    return deviceId;
};
Device_Handle* get_device_handle(int idx){

    return new Device_Handle_hw(idx);
};
void free_device_handle(Device_Handle* handle){
    delete handle;
};