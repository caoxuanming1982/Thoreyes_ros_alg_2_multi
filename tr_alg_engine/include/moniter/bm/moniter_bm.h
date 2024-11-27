#ifndef __DEVICE_MONITER_BM_H__
#define __DEVICE_MONITER_BM_H__

#include "moniter/moniter.h"

class Moniter_bm:public Moniter{
public:
    Moniter_bm(){};
    virtual ~Moniter_bm(){};
    virtual Device_dev_stat get_device_stat(std::shared_ptr<Device_Handle> device);
};


#endif
