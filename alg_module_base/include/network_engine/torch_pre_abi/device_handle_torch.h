#ifndef __NETWORK_DEVICE_HANDLE_TORCH_H__
#define __NETWORK_DEVICE_HANDLE_TORCH_H__
#include "network_engine/device_handle.h"

#include <torch/torch.h>


class Device_Handle_torch:public Device_Handle{
public: 
    torch::Device handle;
    Device_Handle_torch(torch::Device handle_in):handle(handle_in){
        
    };
    virtual ~Device_Handle_torch(){

    };
    virtual int get_device_id();
    virtual int get_card_id();
};


#endif