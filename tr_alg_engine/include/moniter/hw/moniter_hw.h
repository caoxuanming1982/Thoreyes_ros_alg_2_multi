#ifndef __DEVICE_MONITER_HW_H__
#define __DEVICE_MONITER_HW_H__

#include "moniter/moniter.h"

class Moniter_hw:public Moniter{
public:
    Moniter_hw(){};
    virtual ~Moniter_hw(){};
    virtual Device_dev_stat get_device_stat(std::shared_ptr<Device_Handle> device);
};


#endif
