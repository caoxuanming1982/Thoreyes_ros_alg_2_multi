#ifndef __TR_ALG_NODE_BASE_H__
#define __TR_ALG_NODE_BASE_H__
#include<vector>
#include <rclcpp/rclcpp.hpp>
#include<chrono>
#include <shared_mutex>
#include<fstream>
#include<sstream>
#include "tr_interfaces/msg/node_status_report.hpp"

struct Error_status{
    int status=0;
    std::string message="";
};

class Alg_Node_Base:public rclcpp::Node{
protected:
    std::string node_name;
    rclcpp::Publisher<tr_interfaces::msg::NodeStatusReport>::SharedPtr publisher_;
    rclcpp::CallbackGroup::SharedPtr ClientCallBackGroup_;

    Error_status last_error_status;
    rclcpp::TimerBase::SharedPtr timer_;
    long tick_interval_ms=1;

    std::string dev_platform_name="bm";

public:
    Alg_Node_Base(std::string name,std::string dev_platform_name);
    virtual ~Alg_Node_Base();

    void report_state();

    void set_error_data(int status,std::string message);
    void clear_error_data();

};




#endif
