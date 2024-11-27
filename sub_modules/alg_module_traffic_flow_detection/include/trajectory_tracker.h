#ifndef __TRAJECTORY_TRACKER_H__
#define __TRAJECTORY_TRACKER_H__

#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>
#include<cmath>

class Trajectory_Result {
public:
    float score = 0;
    int region_id = -1;
    std::vector<float> labels_socre;
    float dy;
    std::string license = "";
    float license_score;
};

class Centroid_Entity {
public:
    int track_id;
    cv::Point point;
    
    int color = 19; // 0,1,2,3,4,5,6,7,8
    int face_direction = 19; // 9: "面向镜头", 10: "背向镜头"
    int type = 19;

    float color_score;
    float face_direction_score;
    float type_score;

    std::string license = "";
    float license_score = 0;
};

class Trajectory {
public:
    int track_id;                       // 跟踪编号
    std::vector<cv::Point> points;      // 质心轨迹
    int last_update_time = 0;           // 上一次更新时间

    float sum_dy = 0;                   // y轴的变化率
    std::vector<float> labels_socre = { 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0
    }; 
    std::string license = "";
    float license_score = 0;
};

class Trajectory_Tracker {
private:
    int max_record_time = 100;          // 每条轨迹中质心的最大记录次数
    int max_wait_time = 10;             // 更新时间超出该时间的轨迹将被完全删除
    int min_point_for_direction = 2;    // 每条轨迹中最少需要包含多少个点才能推算距离

    std::vector<std::vector<cv::Point>> check_pts;  // 车流检测的检测点
    std::vector<Trajectory> trajectory;

public:
    Trajectory_Tracker();
    ~Trajectory_Tracker();

    void set_max_record_time(int value);
    void set_max_wait_time(int value);
    void set_min_point_for_direction(int value);
    void set_thresh_changelane_distance(float value);

    void init_area(int frame_height, int frame_width, std::vector<std::vector<cv::Point>> boundary_detection_line);
    void update_trajectory(std::vector<Centroid_Entity> entitys);
    bool display_trajectory(int track_id, cv::Mat &image);
    Trajectory_Result get_result(int track_id, float x1, float y1, float x2, float y2);
    int get_trac_region_id(int track_id);
};

#endif