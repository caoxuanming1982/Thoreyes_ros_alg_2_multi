#include "tr_alg_engine/nv/tr_device_manager_nv.h"


std::shared_ptr<Alg_Node_Device_Manager> get_device_manager_node(){
    std::shared_ptr<Alg_Node_Device_Manager_nv> node=std::make_shared<Alg_Node_Device_Manager_nv>();
    std::shared_ptr<Alg_Node_Device_Manager> res=node;
    return res;
};
