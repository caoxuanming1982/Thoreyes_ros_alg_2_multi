#include "tr_alg_engine/bm/tr_check_channel_bm.h"


std::shared_ptr<Alg_Node_Check_Channel> get_check_channel_node(){
    std::shared_ptr<Alg_Node_Check_Channel_bm> node=std::make_shared<Alg_Node_Check_Channel_bm>();
    std::shared_ptr<Alg_Node_Check_Channel> res=node;
    return res;
};
