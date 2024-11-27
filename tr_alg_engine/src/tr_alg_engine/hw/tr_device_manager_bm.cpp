#include "tr_alg_engine/hw/tr_device_manager_hw.h"


std::shared_ptr<Alg_Node_Device_Manager> get_device_manager_node(){
    std::shared_ptr<Alg_Node_Device_Manager_hw> node=std::make_shared<Alg_Node_Device_Manager_hw>();
    std::shared_ptr<Alg_Node_Device_Manager> res=node;
    return res;
};
