#ifndef __TRAJECTORY_TRACKER_H__
#define __TRAJECTORY_TRACKER_H__

#include<iostream>
#include<vector>

class Centroid_Entity {
public:
    int track_id;
    std::pair<float,float> point;
};

class Trajectory {
public:
    int track_id;                           // 跟踪编号
    std::vector<std::pair<float,float>> points;    // 质心轨迹
    int last_update_time = 0;               // 上一次更新时间
};

class Trajectory_Tracker {
private:
    int max_record_time = 100;          // 每条轨迹中质心的最大记录次数
    int max_wait_time = 10;             // 更新时间超出该时间的轨迹将被完全删除
    int min_point_for_direction = 3;    // 每条轨迹中最少需要包含多少个点才能推算距离
    std::vector<Trajectory> trajectory;

public:
    Trajectory_Tracker() {};
    ~Trajectory_Tracker() {};

    void update_trajectory(std::vector<Centroid_Entity> entitys);
    std::pair<float, float> get_trajectory_direction(int track_id);
};

#endif