#ifndef __DUPLICATE_REMOVER_H__
#define __DUPLICATE_REMOVER_H__

#include<iostream>
#include<vector>

class Duplicate_Remover {
public:
    Duplicate_Remover();
    virtual ~Duplicate_Remover();

    virtual bool process(int event_id);
    virtual void update();

    virtual void set_min_repeat_time(int value);
    virtual void set_max_record_time(int value);

protected:    
    int min_repeat_time = 1;                            //事件需要重复报出 n 次才算事件
    int max_record_time = 100;                          //最大记录时间, 超出该时间事件触发id会从已触发中删除, 因此相同事件会重新报出
    std::vector<std::pair<int,int>> triggered_event;    //事件触发id和累计触发时间 事件ID的触发时间
};

#endif