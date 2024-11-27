#include "tr_alg_engine/hw/tr_check_channel_hw.h"


std::shared_ptr<Alg_Node_Check_Channel> get_check_channel_node(){
    std::shared_ptr<Alg_Node_Check_Channel_hw> node=std::make_shared<Alg_Node_Check_Channel_hw>();
    std::shared_ptr<Alg_Node_Check_Channel> res=node;
    return res;
};
