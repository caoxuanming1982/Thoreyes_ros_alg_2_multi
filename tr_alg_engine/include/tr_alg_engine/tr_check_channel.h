#ifndef __TR_CHECK_CHANNEL_H__
#define __TR_CHECK_CHANNEL_H__

#include<vector>
#include <rclcpp/rclcpp.hpp>
#include<chrono>
#include "tr_alg_node_base.h"

#include "tr_alg_interfaces/msg/channel_list.hpp"

class Alg_Node_Check_Channel:public Alg_Node_Base{
protected:
    std::set<std::string> channel_last;

public:
    rclcpp::Publisher<tr_alg_interfaces::msg::ChannelList>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    Alg_Node_Check_Channel(std::string dev_platform_name);

    void check_channel_change();

};

extern "C" std::shared_ptr<Alg_Node_Check_Channel> get_check_channel_node();

#endif
