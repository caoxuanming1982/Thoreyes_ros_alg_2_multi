#ifndef __DEVICE_MONITER_IX_H__
#define __DEVICE_MONITER_IX_H__

#include "moniter/moniter.h"

class Moniter_ix:public Moniter{
public:
    Moniter_ix(){};
    virtual ~Moniter_ix(){};
    virtual Device_dev_stat get_device_stat(std::shared_ptr<Device_Handle> device);
};


#endif
