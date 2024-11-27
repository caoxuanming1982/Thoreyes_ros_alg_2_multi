#ifndef __NETWORK_DEVICE_HANDLE_H__
#define __NETWORK_DEVICE_HANDLE_H__
#include<memory>
class Device_Handle{

public: 
    Device_Handle(){

    };
    virtual ~Device_Handle(){

    };
    virtual int get_device_id()=0;
    virtual int get_card_id()=0;
};

extern "C" Device_Handle* get_device_handle(int idx);
extern "C" void free_device_handle(Device_Handle* handle);
#endif