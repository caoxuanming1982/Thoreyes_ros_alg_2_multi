#ifndef __DEVICE_MONITER_NV_H__
#define __DEVICE_MONITER_NV_H__

#include "moniter/moniter.h"

class Moniter_nv:public Moniter{
public:
    Moniter_nv(){};
    virtual ~Moniter_nv(){};
    virtual Device_dev_stat get_device_stat(std::shared_ptr<Device_Handle> device);
};


#endif
