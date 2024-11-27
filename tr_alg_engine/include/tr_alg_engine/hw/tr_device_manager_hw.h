#ifndef __TR_DEVICE_MANAGER_HW_H__
#define __TR_DEVICE_MANAGER_HW_H__

#include"tr_alg_engine/tr_device_manager.h"

class Alg_Node_Device_Manager_hw:public Alg_Node_Device_Manager{
public:
    Alg_Node_Device_Manager_hw():Alg_Node_Device_Manager("hw"){};
    virtual ~Alg_Node_Device_Manager_hw(){};

};

#endif