#ifndef __NETWORK_DEVICE_HANDLE_BM_H__
#define __NETWORK_DEVICE_HANDLE_BM_H__
#include "network_engine/device_handle.h"

#include<bmruntime.h>
#include<bmruntime_cpp.h>
#include<bmcv_api.h>


class Device_Handle_bm:public Device_Handle{
public: 
    bm_handle_t handle;
    Device_Handle_bm(bm_handle_t handle_in):handle(handle_in){
        
    };
    virtual ~Device_Handle_bm(){

    };
    virtual int get_device_id();
    virtual int get_card_id();
};


#endif