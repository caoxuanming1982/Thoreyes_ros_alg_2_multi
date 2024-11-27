#ifndef __NETWORK_DEVICE_HANDLE_HW_H__
#define __NETWORK_DEVICE_HANDLE_HW_H__
#include "network_engine/device_handle.h"



class Device_Handle_hw:public Device_Handle{
public: 
    aclrtStream handle;
    int deviceId;
    Device_Handle_hw(int deviceId){ 
        this->deviceId=deviceId;
        aclError ret = aclrtSetDevice(deviceId);        
        ret = aclrtCreateStream(&handle);        
    };
    virtual ~Device_Handle_hw(){
        aclError ret = aclrtDestroyStream(stream);
        ret = aclrtResetDevice(deviceId);
    };
    virtual int get_device_id();
    virtual int get_card_id();
};


#endif