#include "tr_alg_node_base.h"
Alg_Node_Base::Alg_Node_Base(std::string name,std::string dev_platform_name) : rclcpp::Node(name)
{
    this->node_name = name;
    this->dev_platform_name=dev_platform_name;
    publisher_ = this->create_publisher<tr_interfaces::msg::NodeStatusReport>("/NodeStatusReport", rclcpp::SensorDataQoS());
    timer_ = this->create_wall_timer(std::chrono::milliseconds(2000), std::bind(&Alg_Node_Base::report_state, this));
    ClientCallBackGroup_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

};
Alg_Node_Base::~Alg_Node_Base(){};

void Alg_Node_Base::report_state()
{
    tr_interfaces::msg::NodeStatusReport msg;
    msg.domain = this->get_namespace();
    msg.name = this->node_name;
    msg.stamp = this->get_clock()->now();
    msg.status = this->last_error_status.status;
    msg.error = this->last_error_status.message;
    publisher_->publish(msg);
};

void Alg_Node_Base::set_error_data(int status, std::string message)
{
    if (this->last_error_status.status == 0 && status != 0)
    {
        this->last_error_status.status = status;
        this->last_error_status.message = message;
        report_state();
    }
};

void Alg_Node_Base::clear_error_data(){
        this->last_error_status.status = 0;
        this->last_error_status.message = "";
};

