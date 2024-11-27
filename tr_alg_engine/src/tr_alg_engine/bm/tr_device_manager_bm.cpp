#include "tr_alg_engine/bm/tr_device_manager_bm.h"


std::shared_ptr<Alg_Node_Device_Manager> get_device_manager_node(){
    std::shared_ptr<Alg_Node_Device_Manager_bm> node=std::make_shared<Alg_Node_Device_Manager_bm>();
    std::shared_ptr<Alg_Node_Device_Manager> res=node;
    return res;
};
