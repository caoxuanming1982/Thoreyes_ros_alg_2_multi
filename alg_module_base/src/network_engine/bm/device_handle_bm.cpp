#include "network_engine/bm/device_handle_bm.h"


int Device_Handle_bm::get_device_id(){
    return bm_get_devid(handle);
};
int Device_Handle_bm::get_card_id(){
    unsigned int card_id=0;
    bm_get_card_id(handle,&card_id);
    return card_id;
};

Device_Handle* get_device_handle(int idx){
    bm_handle_t hd;
    bm_dev_request(&hd,idx);
    return new Device_Handle_bm(hd);
};

void free_device_handle(Device_Handle* handle){
    delete handle;
};