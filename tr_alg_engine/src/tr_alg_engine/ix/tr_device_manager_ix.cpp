#include "tr_alg_engine/ix/tr_device_manager_ix.h"


std::shared_ptr<Alg_Node_Device_Manager> get_device_manager_node(){
    std::shared_ptr<Alg_Node_Device_Manager_ix> node=std::make_shared<Alg_Node_Device_Manager_ix>();
    std::shared_ptr<Alg_Node_Device_Manager> res=node;
    return res;
};
