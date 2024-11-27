#ifndef __TR_DEVICE_MANAGER_BM_H__
#define __TR_DEVICE_MANAGER_BM_H__

#include"tr_alg_engine/tr_device_manager.h"

class Alg_Node_Device_Manager_bm:public Alg_Node_Device_Manager{
public:
    Alg_Node_Device_Manager_bm():Alg_Node_Device_Manager("bm"){};
    virtual ~Alg_Node_Device_Manager_bm(){};

};

#endif