#include "tr_alg_engine/tr_check_channel.h"

Alg_Node_Check_Channel::Alg_Node_Check_Channel(std::string dev_platform_name) : Alg_Node_Base("Check_channel_node",dev_platform_name){
        publisher_ = this->create_publisher<tr_alg_interfaces::msg::ChannelList>("/channel_list_"+dev_platform_name, 2);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(500), std::bind(&Alg_Node_Check_Channel::check_channel_change, this));

}
void Alg_Node_Check_Channel::check_channel_change(){
        std::map<std::string, std::vector<std::string>> topics=get_topic_names_and_types();

        std::set<std::string> channel_current;
        

        for(auto iter=topics.begin();iter!=topics.end();iter++){
            std::string name=iter->first;
//            RCLCPP_INFO(this->get_logger(), "raw channel %s",name.c_str());
            if(name.find("/rv_frames")!=std::string::npos && name.find("/sub_topic")==std::string::npos){
                channel_current.insert(name);
            }
        }
        std::vector<std::string> temp(channel_current.size()+channel_last.size());
        auto iter_start=set_symmetric_difference(channel_current.begin(),channel_current.end(),channel_last.begin(),channel_last.end(),temp.begin());

        auto message=tr_alg_interfaces::msg::ChannelList();
        message.channel_list_string=std::vector<std::string>(channel_current.begin(),channel_current.end());
        if(iter_start-temp.begin()>0)
        {
            message.changed=true;
            this->channel_last=channel_current;
            RCLCPP_INFO(this->get_logger(), "channel change");
            for(auto iter=channel_current.begin();iter!=channel_current.end();iter++){
                RCLCPP_INFO(this->get_logger(), "channel %s",(*iter).c_str());
            }
        }
        else{
            message.changed=false;
  //          RCLCPP_INFO(this->get_logger(), "channel not change");

        }
        this->publisher_->publish(message);


};

