#include "tr_alg_engine/ix/tr_check_channel_ix.h"


std::shared_ptr<Alg_Node_Check_Channel> get_check_channel_node(){
    std::shared_ptr<Alg_Node_Check_Channel_ix> node=std::make_shared<Alg_Node_Check_Channel_ix>();
    std::shared_ptr<Alg_Node_Check_Channel> res=node;
    return res;
};
