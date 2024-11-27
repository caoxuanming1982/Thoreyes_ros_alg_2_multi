#include "tr_alg_engine/nv/tr_check_channel_nv.h"


std::shared_ptr<Alg_Node_Check_Channel> get_check_channel_node(){
    std::shared_ptr<Alg_Node_Check_Channel_nv> node=std::make_shared<Alg_Node_Check_Channel_nv>();
    std::shared_ptr<Alg_Node_Check_Channel> res=node;
    return res;
};
