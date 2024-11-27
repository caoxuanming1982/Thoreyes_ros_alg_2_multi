#include "trajectory_tracker.h"

#define CHECK_PTS_NUM 20

Trajectory_Tracker::Trajectory_Tracker() {

};

Trajectory_Tracker::~Trajectory_Tracker() {

};

void Trajectory_Tracker::set_max_record_time(int value)
{
    if (value >= 0)
        this->max_record_time = value;
    else
        this->max_record_time = 100;
};

void Trajectory_Tracker::set_max_wait_time(int value)
{
    if (value >= 0)
        this->max_wait_time = value;
    else
        this->max_wait_time = 10;
};

void Trajectory_Tracker::set_min_point_for_direction(int value)
{
    if (value >= 0)
        this->min_point_for_direction = value;
    else
        this->min_point_for_direction = 3;
};

/// @brief 初始化区域信息
/// @param frame_height 通道画面的高
/// @param frame_width 通道画面的宽
/// @param boundary_detection_line 检测线区域
void Trajectory_Tracker::init_area(int frame_height, int frame_width, std::vector<std::vector<cv::Point>> boundary_detection_line)
{
    // boundary_detection_line[0] 包含两个点 (x1, y1) 和 (x2, y2)
    this->check_pts.resize(boundary_detection_line.size());

    for (int i = 0; i < boundary_detection_line.size(); ++i) {
        this->check_pts[i].resize(CHECK_PTS_NUM);
        int x1 = boundary_detection_line[i][0].x;
        int y1 = boundary_detection_line[i][0].y;
        int x2 = boundary_detection_line[i][1].x;
        int y2 = boundary_detection_line[i][1].y;
        int w = (x2 - x1) / (CHECK_PTS_NUM-1);
        int h = (y2 - y1) / (CHECK_PTS_NUM-1);
        for (int j = 0; j < CHECK_PTS_NUM; ++j) {
            this->check_pts[i][j].x = x1 + w * j;
            this->check_pts[i][j].y = y1 + h * j;
            // std::cout << this->check_pts[i][j].x << " " << this->check_pts[i][j].y << std::endl;
        }
    }
};

/// @brief 更新轨迹
/// @param entitys 车辆id和目标框中心
void Trajectory_Tracker::update_trajectory(std::vector<Centroid_Entity> entitys)
{
    // 将新的实体放到对应的轨迹中
    for (Centroid_Entity entity : entitys)
    {
        bool flag = true;
        // 实体所对应的轨迹已经存在
        for (Trajectory& trac : this->trajectory) {
            if (trac.track_id != entity.track_id) continue;
            flag = false;
            trac.points.push_back(entity.point); // 增加新的轨迹点
            if (trac.points.size() > this->max_record_time) trac.points.erase(trac.points.begin()); // 超出轨迹点的记录数量

            // 更新车辆信息
            trac.labels_socre[entity.color] += entity.color_score;
            trac.labels_socre[entity.face_direction] += entity.face_direction_score;
            trac.labels_socre[entity.type] += entity.type;

            // 更新车牌信息
            if (trac.license_score < entity.license_score && entity.license.size() > 1) {
                trac.license = entity.license;
                trac.license_score = entity.license_score;
            }

            trac.last_update_time = 0; // 重置轨迹的更新时间
            break;
        }

        // 需要增加新的轨迹
        if (flag) {
            Trajectory new_trajectory;
            new_trajectory.track_id = entity.track_id;
            new_trajectory.points.push_back(entity.point);

            new_trajectory.labels_socre[entity.color] += entity.color_score;
            new_trajectory.labels_socre[entity.face_direction] += entity.face_direction_score;
            new_trajectory.labels_socre[entity.type] += entity.type;

            if (new_trajectory.license_score < entity.license_score && entity.license.size() > 1) {
                new_trajectory.license = entity.license;
                new_trajectory.license_score = entity.license_score;
            }

            this->trajectory.push_back(new_trajectory);
        }
    }
    
    // 删除长时间未更新的轨迹
    std::vector<Trajectory>::iterator trac = this->trajectory.begin();
    for (; trac != this->trajectory.end(); )
    {
        trac->last_update_time += 1;
        if (trac->last_update_time > this->max_wait_time) {
            // 轨迹的上一次更新超出了最大时长
            trac = this->trajectory.erase(trac); // 删除这条轨迹并获取下一条轨迹
        } else {
            // 下一条轨迹
            trac++;
        }
    }
};

/// @brief 获取结果
/// @param track_id 车辆id
Trajectory_Result Trajectory_Tracker::get_result(int track_id, float x1, float y1, float x2, float y2)
{
    Trajectory_Result result;
    result.score = 0;

    Trajectory* trac = nullptr;
    for (int i = 0; i < this->trajectory.size(); ++i) {
        if (this->trajectory[i].track_id == track_id) {
            trac = &this->trajectory[i];
            break;
        }
    }
    
    if (trac == nullptr) return result;                                     // 没有对应目标的轨迹
    if (trac->points.size() < this->min_point_for_direction) return result; // 对应目标的轨迹中小于n个点 无法分辨
    
    for (int i = 0; i < this->check_pts.size(); ++i) {
        for (int j = 0; j < this->check_pts[i].size(); ++j) {
            int _x = this->check_pts[i][j].x;
            int _y = this->check_pts[i][j].y;
            if (_x > x1 && _x < x2 && _y > y1 && _y < y2) {
                result.region_id = i;
                result.score = 1;
                result.labels_socre = trac->labels_socre;
                result.dy = trac->sum_dy;
                result.license = trac->license;
                result.license_score = trac->license_score;
                return result;
            }
        }
    }
    return result;
};

/// @brief 可视化检测结果
/// @param track_id 车辆id
/// @param image 画面
bool Trajectory_Tracker::display_trajectory(int track_id, cv::Mat &image)
{
    Trajectory* trac = nullptr;
    for (int i = 0; i < this->trajectory.size(); ++i) {
        if (this->trajectory[i].track_id == track_id) {
            trac = &this->trajectory[i];
            break;
        }
    }
    if (trac == nullptr) return 0; // 没有对应目标的轨迹
    if (trac->points.size() < this->min_point_for_direction) return 0; // 对应目标的轨迹中小于3个点 无法分辨

    for (int i = 1; i < trac->points.size(); ++i) {
        // cv::Point pt1(trac->points[i-1].first, trac->points[i-1].second);
        // cv::Point pt2(trac->points[i].first, trac->points[i].second);
        cv::Point pt1 = trac->points[i-1];
        cv::Point pt2 = trac->points[i];
        cv::line(image, pt1, pt2, cv::Scalar(int(255*((float)i/(float)trac->points.size())), 0, 0), 3);
    }

    return true;
};
