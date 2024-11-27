/*
 * @Author: wengjie sun sunwengjie@qy666.com.cn
 * @Date: 2024-05-06 14:51:15
 * @LastEditors: wengjie sun sunwengjie@qy666.com.cn
 * @LastEditTime: 2024-07-19 10:07:43
 * @FilePath: /test_data/home/vensin/workspace_sun/alg_module_traffic_flow_detection/src/duplicate_remover.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "duplicate_remover.h"

Duplicate_Remover::Duplicate_Remover()
{

};

Duplicate_Remover::~Duplicate_Remover()
{

};

void Duplicate_Remover::set_min_repeat_time(int value)
{
    if (value <= 0) {
        this->min_repeat_time = 0;
    } else {
        this->min_repeat_time = value;
    }
    return;
};

void Duplicate_Remover::set_max_record_time(int value)
{
    if (value <= 0) {
        this->max_record_time = 0;
    } else {
        this->max_record_time = value;
    }
    return;
};

/// @brief 车辆ID是新ID
/// @param event_id 
bool Duplicate_Remover::process(int event_id)
{
    int i = 0;
    for (; i < this->triggered_event.size(); ++i) { // 判断目标id是否已经发生的事件中
        if (this->triggered_event[i].first == event_id) {
            break;
        }
    }

    if (i >= this->triggered_event.size()) {        // 如果不在, 需要新增事件
        this->triggered_event.push_back(std::pair(event_id, 0));
        return true;
    } else {                                        // 事件已经报出
        return false;
    }
};

/// @brief 更新所有已发生事件的时间
void Duplicate_Remover::update()
{
    std::vector<std::pair<int,int>>::iterator event = this->triggered_event.begin();
    for (; event != this->triggered_event.end(); ) {
        event->second++;
        if (event->second > this->max_record_time) { // 将超出记录时间的时间移除
            event = this->triggered_event.erase(event);
        } else {
            event++;
        }
    }
    return;
};
