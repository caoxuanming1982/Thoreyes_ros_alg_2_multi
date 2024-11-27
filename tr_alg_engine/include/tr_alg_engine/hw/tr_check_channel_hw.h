#ifndef __TR_CHECK_CHANNEL_HW_H__
#define __TR_CHECK_CHANNEL_HW_H__

#include "tr_alg_engine/tr_check_channel.h"

class Alg_Node_Check_Channel_hw:public Alg_Node_Check_Channel{

public:

    Alg_Node_Check_Channel_hw():Alg_Node_Check_Channel("hw"){};
    virtual ~Alg_Node_Check_Channel_hw(){};

};

#endif
