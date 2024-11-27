#ifndef __TR_DEVICE_MANAGER_IX_H__
#define __TR_DEVICE_MANAGER_IX_H__

#include"tr_alg_engine/tr_device_manager.h"

class Alg_Node_Device_Manager_ix:public Alg_Node_Device_Manager{
public:
    Alg_Node_Device_Manager_ix():Alg_Node_Device_Manager("ix"){};
    virtual ~Alg_Node_Device_Manager_ix(){};

};

#endif