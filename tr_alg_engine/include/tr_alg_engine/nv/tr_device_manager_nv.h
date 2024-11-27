#ifndef __TR_DEVICE_MANAGER_NV_H__
#define __TR_DEVICE_MANAGER_NV_H__

#include"tr_alg_engine/tr_device_manager.h"

class Alg_Node_Device_Manager_nv:public Alg_Node_Device_Manager{
public:
    Alg_Node_Device_Manager_nv():Alg_Node_Device_Manager("nv"){};
    virtual ~Alg_Node_Device_Manager_nv(){};

};

#endif