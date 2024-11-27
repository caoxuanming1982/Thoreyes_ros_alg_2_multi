#include "alg_module_helmet_detect_in_region.h"

#include <iostream>
#include <fstream>

cv::Mat drawText(cv::Mat image, int x, int y, string text)
{
    cv::Point point(x, y);
    cv::putText(image, text, point, 1, 2, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
    return image;
}

cv::Mat drawBox(cv::Mat image, int x1, int y1, int x2, int y2, cv::Scalar color = cv::Scalar(255, 127, 255))
{
    // cv::Scalar(255, 255, 255
    cv::Rect box(x1, y1, x2 - x1, y2 - y1);
    cv::rectangle(image, box, color, 2);
    return image;
}

float range_thred(float input, float min, float max)
{
    float val = 0;
    if (input < min)
        val = min;
    else
        val = input;

    if (val > max)
        val = max;

    return val;
}

float iou(Result_item_yolo_sample *box1, Result_item_yolo_sample *box2)
{
    float x1 = std::max(box1->x1, box2->x1);     // left
    float y1 = std::max(box1->y1, box2->y1);     // top
    float x2 = std::min((box1->x2), (box2->x2)); // right
    float y2 = std::min((box1->y2), (box2->y2)); // bottom
    if (x1 >= x2 || y1 >= y2)
        return 0;
    float over_area = (x2 - x1) * (y2 - y1);
    float box1_w = box1->x2 - box1->x1;
    float box1_h = box1->y2 - box1->y1;
    float box2_w = box2->x2 - box2->x1;
    float box2_h = box2->y2 - box2->y1;
    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);
    return iou;
}
float iou(Result_item_yolo_sample &box1, Result_item_yolo_sample &box2)
{
    float x1 = std::max(box1.x1, box2.x1);     // left
    float y1 = std::max(box1.y1, box2.y1);     // top
    float x2 = std::min((box1.x2), (box2.x2)); // right
    float y2 = std::min((box1.y2), (box2.y2)); // bottom
    if (x1 >= x2 || y1 >= y2)
        return 0;
    float over_area = (x2 - x1) * (y2 - y1);
    float box1_w = box1.x2 - box1.x1;
    float box1_h = box1.y2 - box1.y1;
    float box2_w = box2.x2 - box2.x1;
    float box2_h = box2.y2 - box2.y1;
    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);
    return iou;
};


float cos_distance(vector<float> a, vector<float> b)
{
    int length = a.size();
    if (length > b.size())
        length = b.size();
    float temp1 = 0;
    float temp2 = 0;
    float temp3 = 0;
    for (int i = 0; i < length; i++)
    {
        temp1 += a[i] * b[i];
        temp2 += a[i] * a[i];
        temp3 += b[i] * b[i];
    }
    return temp1 / (sqrt(temp2) * sqrt(temp3));
};

void roi_pooling(Output &net_output_feature, vector<Result_item_yolo_sample> &output, int img_h, int img_w)
{
    float *features = (float *)net_output_feature.data.data();

    int f_c = net_output_feature.shape[0];
    int f_h = net_output_feature.shape[1];
    int f_w = net_output_feature.shape[2];
    float factor_h = 1.0 * f_h / img_h;
    float factor_w = 1.0 * f_w / img_w;
    int f_size = f_h * f_w;
    for (int i = 0; i < output.size(); i++)
    {
        output[i].feature.clear();
        int x2 = int(output[i].x2 * factor_w + 1);
        int y2 = int(output[i].y2 * factor_h + 1);
        int x1 = int(output[i].x1 * factor_w);
        int y1 = int(output[i].y1 * factor_h);
        float sub = (y2 - y1) * (x2 - x1);
        output[i].feature.resize(f_c);
        for (int c = 0; c < f_c; c++)
        {
            float val = 0;
            int offset_c = c * f_size;
            for (int h = y1; h < y2; h++)
            {
                int offset_h = h * f_w;
                for (int w = x1; w < x2; w++)
                {
                    val += features[w + offset_h + offset_c];
                }
            }
            if (isinf(val))
            {
                val = 1.8e17;
            }
            else if (isinf(val) == -1)
            {
                val = -1.8e17;
            }
            else if (isnan(val))
            {
                val = 0;
            }
            else
            {
                val = val / sub;
                if (val > 1.8e17)
                    val = 1.8e17;
                else if (val < -1.8e17)
                    val = -1.8e17;
            }
            output[i].feature[c] = val;
        }
    }
};

bool sort_score(Result_item_yolo_sample &box1, Result_item_yolo_sample &box2)
{
    return (box1.score > box2.score);
};

inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
};

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
};

void nms_yolo(Output &net_output, vector<Result_item_yolo_sample> &output, vector<int> class_filter, float threshold_score = 0.25, float threshold_iou = 0.45)
{
    float *input = (float *)net_output.data.data();

    // input: x1, y1, x2, y2, conf, cls

    int dim1 = net_output.shape[1];
    int dim2 = net_output.shape[2];
    vector<Result_item_yolo_sample> result;
    float threshold_score_stage_1 = threshold_score * 0.77;
    for (int k = 0, i = 0; k < dim1; k++, i += dim2)
    {
        float obj_conf = input[i + 9];
        obj_conf = sigmoid(obj_conf);
        if (obj_conf > threshold_score_stage_1)
        {
            Result_item_yolo_sample item;
            float max_class_conf = input[i + 10];
            int max_class_id = 0;
            for (int j = 1; j < dim2 - 10; j++)
            {
                if (input[i + 10 + j] > max_class_conf)
                {
                    max_class_conf = input[i + 10 + j];
                    max_class_id = j;
                }
            }
            max_class_conf = obj_conf * sigmoid(max_class_conf);
            if (max_class_conf > threshold_score_stage_1)
            {

                float cx = (sigmoid(input[i + 5]) * 2 + input[i]) * input[i + 4];
                float cy = (sigmoid(input[i + 6]) * 2 + input[i + 1]) * input[i + 4];
                float w = sigmoid(input[i + 7]) * 2;
                w = w * w * input[i + 2];
                float h = sigmoid(input[i + 8]) * 2;
                h = h * h * input[i + 3];
                float threshold_score_stage_2 = threshold_score * 0.77;
                if (abs(cx - 320) - 160 > 0)
                    threshold_score_stage_2 = threshold_score * (0.67 + 0.4 * w / (abs(cx - 320) - 160));
                if (abs(cy - 176) - 88 > 0)
                    threshold_score_stage_2 = threshold_score * (0.67 + 0.4 * h / (abs(cy - 176) - 88));
                if (threshold_score_stage_2 > threshold_score * 1.2)
                    threshold_score_stage_2 = threshold_score * 1.2;

                if (max_class_conf < threshold_score_stage_2)
                    continue;

                item.x1 = cx - w / 2;
                item.y1 = cy - h / 2;
                item.x2 = cx + w / 2;
                item.y2 = cy + h / 2;

                item.score = max_class_conf;
                item.class_id = max_class_id;
                item.x1 += item.class_id * 4096;
                item.x2 += item.class_id * 4096;
                if (class_filter.size() > 0)
                {
                    if (find(class_filter.begin(), class_filter.end(), max_class_id) != class_filter.end())
                    {
                        result.push_back(item);
                    }
                }
                else
                {
                    result.push_back(item);
                }
            }
        }
    }
    output.clear();
    if (result.size() <= 0)
        return;

    while (result.size() > 0)
    {
        std::sort(result.begin(), result.end(), sort_score);
        output.push_back(result[0]);
        for (int i = 0; i < result.size() - 1; i++)
        {
            float iou_value = iou(result[0], result[i + 1]);
            if (iou_value > threshold_iou)
            {
                result.erase(result.begin() + i + 1);
                i -= 1;
            }
        }
        result.erase(result.begin());
    }
    vector<Result_item_yolo_sample>::iterator iter = output.begin();
    for (; iter != output.end(); iter++)
    {
        iter->x1 -= iter->class_id * 4096;
        iter->x2 -= iter->class_id * 4096;
    }

    return;
};

Duplicate_remover::Duplicate_remover(){};

Duplicate_remover::~Duplicate_remover(){};

void Duplicate_remover::set_min_interval(int interval)
{
    this->min_interval = interval;
    if (this->min_interval < 1)
        this->min_interval = 1;
    if (this->max_interval < this->min_interval)
        this->max_interval = this->min_interval;
};

void Duplicate_remover::set_max_interval(int interval)
{
    this->max_interval = interval;
    if (this->max_interval < this->min_interval)
        this->max_interval = this->min_interval;
};

void Duplicate_remover::set_accept_sim_thres(float thres)
{
    this->accept_sim_thres = thres;
    if (this->accept_sim_thres > 0.99)
        this->accept_sim_thres = 0.99;
    if (this->accept_sim_thres < -1)
        this->accept_sim_thres = -1;
};

void Duplicate_remover::set_trigger_sim_thres(float thres)
{
    this->trigger_sim_thres = thres;
    if (this->trigger_sim_thres > 0.99)
        this->trigger_sim_thres = 0.99;
    if (this->trigger_sim_thres < -1)
        this->trigger_sim_thres = -1;
};

void Duplicate_remover::set_iou_thres(float thres)
{
    this->iou_thres = thres;
};

std::vector<std::pair<int, float>> Duplicate_remover::check_score(std::vector<Result_item_yolo_sample> &result)
{
    std::vector<std::pair<int, float>> res;
    for (int i = 0; i < result.size(); i++)
    {
        Result_item_yolo_sample *item_1 = &(result[i]);
        int last_item_idx = -1;
        float last_sim_score = -1;
        for (int j = 0; j < last_accept_res.size(); j++)
        {
            Result_item_yolo_sample *item_2 = &(last_accept_res[j].first);
            if (item_1->class_id != item_2->class_id)
                continue;
            float iou_value = iou(item_1, item_2);
            if (iou_value >= iou_thres)
            {
                float sim_score = cos_distance(item_1->feature, item_2->feature);
                if (sim_score > last_sim_score)
                {
                    last_item_idx = j;
                    last_sim_score = sim_score;
                }
            }
        }
        res.push_back(std::make_pair(last_item_idx, last_sim_score));
    }
    return res;
};

std::vector<Result_item_yolo_sample> Duplicate_remover::process(std::vector<Result_item_yolo_sample> result)
{
    for (int i = 0; i < last_accept_res.size(); i++)
    {
        last_accept_res[i].second++;
    }
    for (auto iter = last_accept_res.begin(); iter != last_accept_res.end(); iter++)
    {
        if ((*iter).second > max_interval + 10)
        {
            iter = last_accept_res.erase(iter);
            iter--;
        }
    }

    std::vector<std::pair<int, float>> sim_score = check_score(result);
    std::set<int> filted_res_idx;
    for (int i = 0; i < result.size(); i++)
    {
        std::pair<int, float> idx_score = sim_score[i];
        if (idx_score.first < 0 || idx_score.second < trigger_sim_thres)
        {
            filted_res_idx.insert(i);
        }
    }
    for (int i = 0; i < result.size(); i++)
    {
        if (filted_res_idx.find(i) != filted_res_idx.end())
        {
            continue;
        }
        std::pair<int, float> idx_score = sim_score[i];
        if (idx_score.second < accept_sim_thres)
        {
            int interval = (idx_score.second - trigger_sim_thres) / (accept_sim_thres - trigger_sim_thres) * (max_interval - min_interval);
            if (last_accept_res[idx_score.first].second > interval)
            {
                filted_res_idx.insert(i);
            }
        }
        else
        {
            if (last_accept_res[idx_score.first].second >= max_interval)
            {
                filted_res_idx.insert(i);
            }
        }
    }
    std::vector<Result_item_yolo_sample> filted_res;
    for (int i = 0; i < result.size(); i++)
    {
        if (filted_res_idx.find(i) == filted_res_idx.end())
        {
            continue;
        }

        std::pair<int, float> idx_score = sim_score[i];
        if (idx_score.first < 0)
        {
            //			cout<<idx_score.first<<"\t"<<idx_score.second<<"\t"<<"-1\t";
            last_accept_res.push_back(std::make_pair(result[i], 0));
        }
        else
        {
            //			cout<<idx_score.first<<"\t"<<idx_score.second<<"\t"<<last_accept_res[idx_score.first].second<<"\t";
            last_accept_res[idx_score.first] = std::make_pair(result[i], 0);
        }
        filted_res.push_back(result[i]);
        //		cout<<endl;
    }

    return filted_res;
};

std::vector<std::pair<std::string, std::vector<std::pair<float, float>>>> load_boundary_from_file(std::string path)
{

    std::ifstream infile;
    infile.open(path);
    if (!infile.is_open())
    {
        printf("open file failure!\n");
    }
    std::vector<std::pair<std::string, std::vector<std::pair<float, float>>>> result;
    while (!infile.eof())
    {
        string line;
        infile >> line;
        std::vector<std::string> items;
        supersplit(line, items, ":");
        if (items.size() <= 1)
            continue;
        string boundary_name = items[0];

        std::vector<std::string> items1;
        supersplit(items[1], items1, ";");
        std::vector<std::pair<float, float>> points;
        for (int i = 0; i < items1.size(); i++)
        {
            std::vector<std::string> xy_item;
            supersplit(items1[i], xy_item, ",");
            float x = stringToNum<float>(xy_item[0]);
            float y = stringToNum<float>(xy_item[1]);
            points.push_back(std::make_pair(x, y));
        }
        result.push_back(std::make_pair(boundary_name, points));
    }
    infile.close();
    return result;
}
void Channel_data_helmet_detect_in_region::set_timestamps(std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> timestamps)
{
    // 清空现有的 timestamps_（如果需要）
    timestamps_.clear();

    // 遍历输入的 timestamps
    for (const auto& entry : timestamps)
    {
        const std::string& time_name = entry.first;
        const auto& time_pairs = entry.second;

        // 将每个时间对添加到 timestamps_
        timestamps_.emplace_back(time_name, time_pairs);
    }
}



void Channel_data_helmet_detect_in_region::set_boundarys(std::vector<std::pair<std::string, std::vector<std::pair<float, float>>>> boundarys)
{

    for (int mask_idx = 0; mask_idx < boundarys.size(); mask_idx++)
    {
        string region_name = boundarys[mask_idx].first;
        //std::cout<<region_name<<std::endl;
        std::vector<std::pair<float, float>> boundary = boundarys[mask_idx].second;
        if (accepted_boundary_name.find(region_name) == accepted_boundary_name.end())
        {
            continue;
        }
        this->mask_idx2mask_name[mask_idx] = region_name;

        std::vector<cv::Point2f> boundary_;
        for (int i = 0; i < boundary.size(); i++)
        {
            boundary_.push_back(cv::Point2f(boundary[i].first, boundary[i].second));
        }
        this->boundarys_[mask_idx] = boundary_;
        //std::cout<<this->boundarys_[mask_idx]<<std::endl;
    }
}

void Channel_data_helmet_detect_in_region::set_need_remove_duplicate(bool flag)
{
    this->need_remove_duplicate = flag;
    this->need_feature = flag;
};

void Channel_data_helmet_detect_in_region::set_need_features(bool flag)
{
    this->need_feature = flag;
};

void Channel_data_helmet_detect_in_region::add_boundary(std::string region_name, std::vector<std::pair<float, float>> boundary)
{
    int mask_idx = boundarys_.size();
    if (accepted_boundary_name.find(region_name) == accepted_boundary_name.end())
    {
        return;
    }
    mask_idx2mask_name[mask_idx] = region_name;
    std::vector<cv::Point2f> boundary_;
    for (int i = 0; i < boundary.size(); i++)
    {
        boundary_.push_back(cv::Point2f(boundary[i].first, boundary[i].second));
    }
    boundarys_[mask_idx] = boundary_;
};

void Channel_data_helmet_detect_in_region::init_buffer(int width, int height)
{
    grid_width = int(grid_factor * width + 1);
    grid_height = int(grid_factor * height + 1);
    //	cout<<grid_width<<"\t"<<grid_height<<endl;
    for (auto iter = boundarys_.begin(); iter != boundarys_.end(); iter++)
    {
        int mask_idx = iter->first;
        //	for(int mask_idx=0;mask_idx<boundarys_.size();mask_idx++){
        mask_grids[mask_idx] = cv::Mat::zeros(grid_height, grid_width, CV_16UC1);
        std::vector<cv::Point2f> boundary_ = iter->second; // boundarys_[mask_idx];

        std::vector<std::vector<cv::Point>> temp_boundary;
        std::vector<cv::Point> temp_bondary_;

        for (int i = 0; i < boundary_.size(); i++)
        {
            cv::Point2f point = boundary_[i];
            temp_bondary_.push_back(cv::Point(int(point.x * (grid_width - 1)), int(point.y * (grid_height - 1))));
        }

        temp_boundary.push_back(temp_bondary_);

        grid_temp[mask_idx] = cv::Mat::zeros(grid_height, grid_width, CV_16UC1);

        cv::fillPoly(mask_grids[mask_idx], temp_boundary, cv::Scalar(1));
        mask_fg_cnt[mask_idx] = cv::sum(mask_grids[mask_idx])[0];

        if (need_count_check)
        {
            count_grids[mask_idx] = cv::Mat::zeros(grid_height, grid_width, CV_16UC1);
        }
    }
};

void Channel_data_helmet_detect_in_region::region_check(vector<Result_item_yolo_sample> detect_result)
{

    for (int i = 0; i < detect_result.size(); i++)
    {
        Result_item_yolo_sample &item = detect_result[i];
        cv::Rect roi = cv::Rect(item.x1 * grid_factor - 0.6, item.y1 * grid_factor - 0.6, (item.x2 - item.x1) * grid_factor + 0.6, (item.y2 - item.y1) * grid_factor + 0.6);
        if (roi.x < 0)
            roi.x = 0;
        if (roi.y < 0)
            roi.y = 0;
        if (roi.x + roi.width > grid_width)
            roi.width = grid_width - roi.x;
        if (roi.y + roi.height > grid_height)
            roi.height = grid_height - roi.y;

        for (auto iter = mask_grids.begin(); iter != mask_grids.end(); iter++)
        {
            string mask_name = mask_idx2mask_name[iter->first];
            if (region_depended_class.find(mask_name) != region_depended_class.end())
            {
                auto class_idxes = region_depended_class.find(mask_name)->second;
                if (class_idxes.size() == 0 || class_idxes.find(item.class_id) != class_idxes.end())
                {

                    cv::Mat roi_mask = iter->second(roi);
                    cv::Mat roi_temp = grid_temp[iter->first](roi);
                    roi_temp += roi_mask;
                    

                }
            }
        }
    }
    for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
    {
        cv::Mat temp = grid_temp[iter->first] > 0;
        temp.convertTo(grid_temp[iter->first], grid_temp[iter->first].type(), 1 / 255.0);
    }
};

void Channel_data_helmet_detect_in_region::count_add()
{
    for (auto iter = mask_grids.begin(); iter != mask_grids.end(); iter++)
    {
        if (need_count_check)
        {
            if (check_exist)
            {
                cv::Mat temp = count_grids[iter->first] + grid_temp[iter->first];
                cv::Mat mask = grid_temp[iter->first] > 0;
                cv::Mat mask1 = grid_temp[iter->first] <= 0;
                cv::Mat mask_t, mask_t1;
                (mask).convertTo(mask_t, temp.type(), 1 / 255.0);
                (mask1).convertTo(mask_t1, temp.type(), 1 / 255.0);
                temp = (temp).mul(mask_t);
                count_grids[iter->first] = temp + mask_t1 * (-2);
            }
            else
            {
                count_grids[iter->first] += iter->second;
                cv::Mat mask = grid_temp[iter->first] <= 0;
                cv::Mat mask_t;
                (mask).convertTo(mask_t, count_grids[iter->first].type(), 1 / 255.0);
                count_grids[iter->first] = count_grids[iter->first].mul(mask_t);
            }
        }

    }
};

void Channel_data_helmet_detect_in_region::count_check()
{

    if (need_count_check)
    {
        if (check_exist)
        {
            for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
            {
                cv::Mat temp = count_grids[iter->first] > check_count_thres;
                temp.convertTo(iter->second, iter->second.type(), 1 / 255.0);
                //				iter->second=count_grids[iter->first]>check_count_thres;
            }
        }
        else
        {
            for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
            {
                cv::Mat temp = count_grids[iter->first] > check_count_thres;
                temp.convertTo(iter->second, iter->second.type(), 1 / 255.0);
                //				iter->second=(count_grids[iter->first]>check_count_thres);
            }
        }
    }
    else
    {
        if (check_exist)
        {
            for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
            {
                cv::Mat temp = iter->second > 0;
                temp.convertTo(iter->second, iter->second.type(), 1 / 255.0);

                //				iter->second=iter->second>0;
            }
        }
        else
        {
            for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
            {
                cv::Mat mask = grid_temp[iter->first] <= 0;
                cv::Mat mask_t;
                (mask).convertTo(mask_t, grid_temp[iter->first].type(), 1 / 255.0);

                iter->second = mask_t.mul(mask_grids[iter->first]) > 0;
            }
        }
    }
};

vector<Result_item_yolo_sample> Channel_data_helmet_detect_in_region::get_result(vector<Result_item_yolo_sample> detect_result)
{
    vector<Result_item_yolo_sample> result;
    if (check_exist)
    {

        for (int i = 0; i < detect_result.size(); i++)
        {
            Result_item_yolo_sample &item = detect_result[i];

            cv::Rect roi = cv::Rect(item.x1 * grid_factor + 0.5, item.y1 * grid_factor + 0.5, (item.x2 - item.x1) * grid_factor + 0.5, (item.y2 - item.y1) * grid_factor + 0.5);
            if (roi.x < 0)
                roi.x = 0;
            if (roi.y < 0)
                roi.y = 0;
            if (roi.x + roi.width > grid_width)
                roi.width = grid_width - roi.x;
            if (roi.y + roi.height > grid_height)
                roi.height = grid_height - roi.y;
            float score = 0;
            int region_idx = -1;
            for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
            {
                string mask_name = mask_idx2mask_name[iter->first];
                //cout<<mask_name<<endl;
                if (region_depended_class.find(mask_name) != region_depended_class.end())
                {
                    auto class_idxes = region_depended_class.find(mask_name)->second;

                    if (class_idxes.size() == 0 || class_idxes.find(item.class_id) != class_idxes.end())
                    {

                        cv::Mat roi_temp = grid_temp[iter->first](roi);
            
                        // Print debugging information
                        //std::cout << "ROI: (" << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height << ")\n";
                        //std::cout << "ROI Temp Mat Size: " << roi_temp.size() << "\n";
                        int sum_value = cv::sum(roi_temp)[0];
                        //std::cout << "Sum Value: " << sum_value << "\n";
                        int box_size = (item.x2 - item.x1) * (item.y2 - item.y1) * grid_factor * grid_factor;
                        //std::cout << "Box Size: " << box_size << "\n";
                        float temp_score = 1.0 * sum_value / box_size * item.score;
                        //std::cout << "temp_score: " << temp_score << "\n";
                        //					float temp_score=cv::sum(roi_temp)[0]*1.0/(item.x2-item.x1)/(item.y2-item.y1)*item.score;
                        if (temp_score > score)
                        {
                            score = temp_score;
                            region_idx = iter->first;
                        }
                    }
                }
            }

            if (score > 0.75)
            {
                Result_item_yolo_sample res;
                res.temp_idx = i;
                res.x1 = item.x1;
                res.y1 = item.y1;
                res.x2 = item.x2;
                res.y2 = item.y2;
                res.class_id = item.class_id;
                res.region_idx = region_idx;
                res.score = score;
                res.contour = vector<std::pair<float, float>>();
                res.contour.push_back(std::make_pair(res.x1, res.y1));
                res.contour.push_back(std::make_pair(res.x2, res.y1));
                res.contour.push_back(std::make_pair(res.x2, res.y2));
                res.contour.push_back(std::make_pair(res.x1, res.y2));
                result.push_back(res);
            }
        }
    }
    else
    {
        for (auto iter = grid_temp.begin(); iter != grid_temp.end(); iter++)
        {
            int fg_cng = cv::sum(iter->second)[0];
            if (fg_cng < check_sensitivity_thres * mask_fg_cnt[iter->first] || fg_cng < 4)
            {
                continue;
            }
            vector<vector<cv::Point>> contours;
            cv::Mat temp = grid_temp[iter->first] > 0;
            cv::findContours(temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            for (int i = 0; i < contours.size(); i++)
            {
                cv::Rect rect = cv::boundingRect(contours[i]);
                Result_item_yolo_sample res;
                res.x1 = (rect.x + 0.5) / grid_factor;
                res.y1 = (rect.y + 0.5) / grid_factor;
                res.x2 = (rect.x + rect.width + 0.5) / grid_factor;
                res.y2 = (rect.y + rect.height + 0.5) / grid_factor;
                res.class_id = iter->first;
                res.temp_idx = -1;
                res.region_idx = iter->first;
                vector<cv::Point> &contour = contours[i];
                //			approxPolyDP(contours[i],contour,3,true);

                res.score = cv::contourArea(contour) / mask_fg_cnt[iter->first];
                if (res.score < 0.1)
                    continue;
                res.contour = vector<std::pair<float, float>>();
                for (int j = 0; j < contour.size(); j++)
                {
                    cv::Point &point = contour[j];
                    res.contour.push_back(std::make_pair((point.x + 0.5) / grid_factor, (point.y + 0.5) / grid_factor));
                }
                result.push_back(res);
            }
        }
    }


  /* for (const auto& item : result)
    {
        std::cout << "类 ID: " << item.class_id 
                  << ", score: " << item.score 
                  << ", 边界框: [" 
                  << item.x1 << ", " << item.y1 << ", " 
                  << item.x2 << ", " << item.y2 << "]" 
                  << std::endl;
    }*/
    return result;
};

vector<Result_item_yolo_sample> Channel_data_helmet_detect_in_region::find_work_region(
    vector<Result_item_yolo_sample> &detect_res)
{
    vector<cv::Point> region_points;
    bool has_construction_truck = false;
    vector<Result_item_yolo_sample> result = detect_res;
    for (int i = 0; i < detect_res.size(); i++)
    {
        if (detect_res[i].class_id > 0)
        {
            has_construction_truck = true;
            break;
        }
    }
    if (has_construction_truck == false)
    {
        return result;
    }
    float score = 0;
    bool has_block = false;
    for (int i = 0; i < detect_res.size(); i++)
    {
        if (detect_res[i].class_id == 0)
        {
            auto &temp = detect_res[i];
            float c_x = (temp.x1 + temp.x2) / 2;
            float c_y = (temp.y1 + temp.y2) / 2 + (temp.y2 - temp.y1) * 0.3;
            region_points.push_back(cv::Point(c_x, c_y));
            score += 1;
            has_block = true;
        }
        else
        {
            auto &temp = detect_res[i];
            float c_x = (temp.x1 + temp.x2) / 2;
            float c_y = (temp.y1 + temp.y2) / 2;
            float w = (temp.x2 - temp.x1);
            float h = (temp.y2 - temp.y1);
            region_points.push_back(cv::Point(c_x - w * 0.35, c_y - h * 0.15));
            region_points.push_back(cv::Point(c_x + w * 0.35, c_y - h * 0.15));
            region_points.push_back(cv::Point(c_x + w * 0.35, c_y + h * 0.4));
            region_points.push_back(cv::Point(c_x - w * 0.35, c_y + h * 0.4));
            score += 10;
        }
    }
    if (score > 25)
        score = 25;

    if (score > 13 && has_block)
    {
        vector<cv::Point> work_region_points;
        cv::convexHull(region_points, work_region_points);
        Result_item_yolo_sample res;
        cv::Rect rect = cv::boundingRect(work_region_points);
        res.temp_idx = -1;
        res.x1 = rect.x;
        res.y1 = rect.y;
        res.x2 = rect.x + rect.width;
        res.y2 = rect.y + rect.height;
        res.class_id = -1;
        res.region_idx = -1;
        res.score = score / 25;
        res.contour = vector<std::pair<float, float>>();
        for (int j = 0; j < work_region_points.size(); j++)
        {
            cv::Point &point = work_region_points[j];
            res.contour.push_back(std::make_pair(point.x, point.y));
        }

        result.push_back(res);
    }
    return result;
};

void Channel_data_helmet_detect_in_region::reset_channal_data()
{
    // 清理通道数据
    
}
void Channel_data_helmet_detect_in_region::set_module_data(Module_data &module_data)
{

    class_id2class_name = module_data.class_id2class_name;
    accepted_boundary_name = module_data.accepted_boundary_name;
    region_depended_class = module_data.region_depended_class;
    check_exist = module_data.check_exist;
    need_count_check = module_data.need_count_check;
    check_count_thres = module_data.check_count_thres;
    check_sensitivity_thres = module_data.check_sensitivity_thres;
    check_area_thres = module_data.check_area_thres;
    dupl_rm_accept_sim_thres = module_data.dupl_rm_accept_sim_thres;
    dupl_rm_trigger_sim_thres = module_data.dupl_rm_trigger_sim_thres;
    dupl_rm_iou_thres = module_data.dupl_rm_iou_thres;
    dupl_rm_min_interval = module_data.dupl_rm_min_interval;
    dupl_rm_max_interval = module_data.dupl_rm_max_interval;

    remover.set_accept_sim_thres(dupl_rm_accept_sim_thres);
    remover.set_trigger_sim_thres(dupl_rm_trigger_sim_thres);
    remover.set_iou_thres(dupl_rm_iou_thres);
    remover.set_min_interval(dupl_rm_min_interval);
    remover.set_max_interval(dupl_rm_max_interval);
}

std::string Channel_data_helmet_detect_in_region::decode_tag(Result_item_yolo_sample item)
{
    if (check_exist)
    {
        if(item.class_id==1)
            return "未带头盔";
        else{
            return "for_debug";
        }
    }
    else
    {
        return mask_idx2mask_name[item.class_id] + "丢失或移位";
    }
};

Alg_Module_helmet_detect_in_region::Alg_Module_helmet_detect_in_region() : Alg_Module_Base_private("helmet_detect_in_region")
{ // 参数是模块名，使用默认模块名初始化

    module_data_.accepted_boundary_name.insert("非机动车_道路");
    //module_data_.accepted_time_name.insert("bus_time");

    module_data_.class_id2class_name.insert(std::make_pair(0, "person"));
    module_data_.class_id2class_name.insert(std::make_pair(1, "bicycle"));
    module_data_.class_id2class_name.insert(std::make_pair(2, "car"));
    module_data_.class_id2class_name.insert(std::make_pair(3, "motorcycle"));
    module_data_.class_id2class_name.insert(std::make_pair(4, "bus"));
    module_data_.class_id2class_name.insert(std::make_pair(5, "truck"));
    {
        std::set<int> temp;
        temp.insert(0);
        temp.insert(1);
        temp.insert(2);
        temp.insert(3);
        temp.insert(4);
        temp.insert(5);
        module_data_.region_depended_class.insert(std::make_pair("非机动车_道路", temp));


    }

    module_data_.check_count_thres = 0;   // 1
    module_data_.need_count_check = false; // true
    module_data_.check_exist = true;      // true

    // module_data_.classes;
    module_data_.check_sensitivity_thres = DEFAULT_CHECK_SENSITIVITY_THRES;
    module_data_.dupl_rm_accept_sim_thres = DEFAULT_DUPL_RM_ACCEPT_SIM_THRES;
    module_data_.dupl_rm_trigger_sim_thres = DEFAULT_DUPL_RM_TRIGGER_SIM_THRES;
    module_data_.dupl_rm_iou_thres = DEFAULT_DUPL_RM_IOU_THRES;

    module_data_.dupl_rm_min_interval = DEFAULT_DUPL_RM_MIN_INTERVAL;
    module_data_.dupl_rm_max_interval = DEFAULT_DUPL_RM_MAX_INTERVAL;

};

Alg_Module_helmet_detect_in_region::~Alg_Module_helmet_detect_in_region(){

};

void Alg_Module_helmet_detect_in_region::init_module_data(std::shared_ptr<Module_cfg_Yolo_helmet_detect_in_region> module_cfg)
{
    bool load_param_res = true;
    load_param_res = module_cfg->get_float("check_sensitivity_thres", module_data_.check_sensitivity_thres);
    load_param_res = module_cfg->get_float("dupl_rm_accept_sim_thres", module_data_.dupl_rm_accept_sim_thres);
    load_param_res = module_cfg->get_float("dupl_rm_trigger_sim_thres", module_data_.dupl_rm_trigger_sim_thres);
    load_param_res = module_cfg->get_float("dupl_rm_iou_thres", module_data_.dupl_rm_iou_thres);
    load_param_res = module_cfg->get_int("dupl_rm_min_interval", module_data_.dupl_rm_min_interval);
    load_param_res = module_cfg->get_int("dupl_rm_max_interval", module_data_.dupl_rm_max_interval);
    //加载重叠区域阈值参数
    load_param_res = module_cfg->get_float("check_area_thres", module_data_.check_area_thres);
}

bool Alg_Module_helmet_detect_in_region::init_from_root_dir(std::string root_dir)
{
    // break alg_module_helmet_detect_in_region.cpp:424
    bool res = this->load_module_cfg(root_dir + "/cfgs/" + this->node_name + "/module_cfg.xml");

    if (res == false)
    {
        std::cout << model_name << "  something wrong when load_module_cfg" << std::endl;
        return res;
    }

    std::shared_ptr<Module_cfg_base> module_cfg = this->get_module_cfg(); // 获取模块配置
    bool load_param_res = true;

    auto algo_module_cfg =
        std::dynamic_pointer_cast<Module_cfg_Yolo_helmet_detect_in_region>(module_cfg);
    if (!algo_module_cfg)
    {
        return false;
    }

    init_module_data(algo_module_cfg);

    // 如果文件中有运行频率的字段，则使用文件中设定的频率
    int tick_interval;
    load_param_res = module_cfg->get_int("tick_interval", tick_interval);
    if (load_param_res)
        this->tick_interval_ms = tick_interval_ms;
    else
        this->tick_interval_ms = DEFAULT_TICK_INTERVAL_MS;

    load_param_res = module_cfg->get_string("model_path", this->model_path); // 加载模型路径
    if (!load_param_res)
        throw Alg_Module_Exception("no model_path in cfgs", this->node_name, Alg_Module_Exception::Stage::check);
    load_param_res = module_cfg->get_string("model_name", this->model_name); // 加载模型名称
    if (!load_param_res)
        throw Alg_Module_Exception("no model_name in cfgs", this->node_name, Alg_Module_Exception::Stage::check);
    load_param_res = module_cfg->get_string("model_cfg_path", this->model_cfg_path); // 加载模型路径
    if (!load_param_res)
        throw Alg_Module_Exception("no model_cfg_path in cfgs", this->node_name, Alg_Module_Exception::Stage::check);

    // break alg_module_lost_object1_detection.cpp:424
    load_param_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name +"/"+ this->model_cfg_path, this->model_name);
    if (!load_param_res)
        throw Alg_Module_Exception("load_model_cfg failed", this->node_name, Alg_Module_Exception::Stage::check);

    // 加载模型
    res = this->load_model(root_dir + "/models/" + this->model_path, this->model_name);
    if (res == false)
    {
        std::cout << model_name << "  something wrong when load_model" << std::endl;
        return res;
    }
        // 加载工人分类模型
    load_param_res = module_cfg->get_string("helmet_model_path", this->helmet_model_path);
    load_param_res = module_cfg->get_string("helmet_model_name", this->helmet_model_name);
    load_param_res = module_cfg->get_string("helmet_model_cfg_path", this->helmet_model_cfg_path);
    if (!load_param_res) throw Alg_Module_Exception("load param in module_cfgs failed", this->node_name, Alg_Module_Exception::Stage::check);

    load_param_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name+"/" + this->helmet_model_cfg_path, this->helmet_model_name);
    if (!load_param_res) throw Alg_Module_Exception("load model_cfgs failed", this->node_name, Alg_Module_Exception::Stage::check);

    load_param_res = this->load_model(root_dir + "/models/" + this->helmet_model_path, this->helmet_model_name);
    if (!load_param_res) throw Alg_Module_Exception("load model failed", this->node_name, Alg_Module_Exception::Stage::check);
    // 加载行人检测需要的参数
    load_param_res &= module_cfg->get_float("helmet_thresh_score", this->helmet_thresh_score);
    load_param_res &= module_cfg->get_float("helmet_thresh_iou", this->helmet_thresh_iou);
    return true;
};
void Channel_data_helmet_detect_in_region::filter_invalid_objects(std::vector<Result_item_yolo_sample> &objects) 
{   
    const float scaleFactor = 0.4f;
    if (this->boundarys_.size() == 0) { // 不存在边界就移除所有结果
        objects.clear();
        return;
    }
    if(objects.size() == 0) return;
    std::vector<Result_item_yolo_sample>::iterator object = objects.begin();
    for (; object != objects.end(); )
    {   
        object->region_idx = -1;
        // 计算归一化坐标
        float normX1 = static_cast<float>(object->x1) / this->_img_w;
        float normY1 = static_cast<float>((scaleFactor*(object->y2-object->y1)+object->y1)) / this->_img_h;
        float normX2 = static_cast<float>(object->x2) / this->_img_w;
        float normY2 = static_cast<float>(object->y2) / this->_img_h;
        std::vector<cv::Point2f> objectPoints = {
                cv::Point2f(normX1, normY1),
                cv::Point2f(normX2, normY1),
                cv::Point2f(normX2, normY2),
                cv::Point2f(normX1, normY2)
        };
        std::vector<float> area(this->boundarys_.size(), 0.0f); 

        for (int i = 0; i < this->boundarys_.size(); ++i) {
            double dist = cv::pointPolygonTest(this->boundarys_[i], 
                cv::Point2f((object->x1 + object->x2) / 2 / this->_img_w, 
                            (object->y1 + object->y2) / 2 / this->_img_h), false);
            
            if (dist >= 0) { // 目标在边界上或边界中
                object->region_idx = i;
                std::vector<cv::Point2f> overlap;
                cv::intersectConvexConvex(boundarys_[i],objectPoints,  overlap);

                if (overlap.empty()) {
                    return; // 如果没有重叠区域
                }

                area[i] = cv::contourArea(overlap); // 将面积添加到对应的区域
            }
        }
        float area_temp = cv::contourArea(objectPoints);//获取检测结果的总面积
        //std::cout << "Area_temp: " << area_temp << std::endl;

        for (size_t i = 0; i < area.size(); ++i) {

            if (area[i] < check_area_thres && area[i] < (area_temp*scaleFactor)) { // 过滤条件小于阈值并且小于自身总面积  过滤掉删除
                std::cout << "删除Area: " << area[i] << std::endl;
                object=objects.erase(object);
            }else{
                object->area=area[i];//保存area到result里面
                //cout<< "Area_temp: " << area_temp <<"object_area"<<object->area<<endl;    //打印出自身总面积和重叠面积
                object++;
            }    
        }
    }

    return;
};
bool Alg_Module_helmet_detect_in_region::detect_engineering_factor(std::string channel_name, std::shared_ptr<Device_Handle> &handle, std::shared_ptr<QyImage> &input_image, std::vector<Result_item_yolo_sample> &result)
{
    std::shared_ptr<Channel_data_helmet_detect_in_region> channel_data = std::dynamic_pointer_cast<Channel_data_helmet_detect_in_region>(this->get_channal_data(channel_name));
    std::shared_ptr<Model_cfg_Yolo_helmet_detect_in_region> model_cfg = std::dynamic_pointer_cast<Model_cfg_Yolo_helmet_detect_in_region>(this->get_model_cfg(this->model_name));
    std::shared_ptr<Module_cfg_Yolo_helmet_detect_in_region> module_cfg = std::dynamic_pointer_cast<Module_cfg_Yolo_helmet_detect_in_region>(this->get_module_cfg());

    //检查参数设置
    std::vector<int> classes;
    // 只检测非机动车
    classes.push_back(3);
    float thresh_iou;
    float thresh_score;
    bool load_res = true;
    //load_res &= module_cfg->get_int_vector("classes", classes);
    load_res &= module_cfg->get_float("thresh_score", thresh_score);
    load_res &= module_cfg->get_float("thresh_iou", thresh_iou);
    if (load_res == false) {
        throw Alg_Module_Exception("Error:\t load module param failed",this->node_name,Alg_Module_Exception::Stage::inference);         //找不到必要的配置参数，检查配置文件是否有对应的字段，检查类型，检测名称
        return false;
    }

    //获取指定的模型实例
    auto net = this->get_model_instance(this->model_name);
    if (net == nullptr) {
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);       //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        return false;
    }

    //判断模型是否已经加载
    auto input_shapes = net->get_input_shapes();
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;
    }

    /////////////////以下为原版本的yolov5模块的代码//////////////////////
    auto input_shape_ = input_shapes[0]; //[channel, height, width]

    // 计算图片尺寸(input_image)和模型输入尺寸(input_shape_)的比例
    float factor1 = input_shape_.dims[3] * 1.0 / input_image->get_width();
    float factor2 = input_shape_.dims[2] * 1.0 / input_image->get_height();


    float factor = factor1 > factor2 ? factor2 : factor1; // 选择较小的比例
    int target_width = input_image->get_width() * factor;        // 图片需要缩放到的目标宽度
    int target_height = input_image->get_height() * factor;      // 图片需要缩放到的目标高度
    
    if((target_width < 8) || (target_width > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " target_width invalid " << target_width << std::endl;
        return false;
    }
    if((target_height < 8) || (target_height > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " target_height invalid " << target_height << std::endl;
        return false;
    }
    if((input_image->get_width() < 8) || (input_image->get_width() > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.width invalid " << input_image->get_width() << std::endl;
        return false;
    }
    if((input_image->get_height() < 8) || (input_image->get_height() > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.height invalid " << input_image->get_height() << std::endl;
        return false;
    }
    
    std::vector<Output> net_output;
    cv::Rect crop_rect;
    crop_rect.x = 0;
    crop_rect.y = 0;
    crop_rect.width = input_image->get_width();
    crop_rect.height= input_image->get_height();

    std::shared_ptr<QyImage> sub_image = input_image->crop_resize_keep_ratio(crop_rect,input_shape_.dims[3],input_shape_.dims[2],0);

    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(sub_image);
    net->forward(net_input, net_output);


    std::vector<Result_item_yolo_sample> detections;
    nms_yolo(net_output[0], detections, classes, thresh_score, thresh_iou);

    for (auto iter = detections.begin(); iter != detections.end(); iter++)
    {
        iter->x1 = (iter->x1 + 0.5) / factor;
        iter->y1 = (iter->y1 + 0.5) / factor;
        iter->x2 = (iter->x2 + 0.5) / factor;
        iter->y2 = (iter->y2 + 0.5) / factor;
    }

    roi_pooling(net_output[1], detections, input_shape_.dims[2] / factor, input_shape_.dims[3] / factor);

    // 原模块需要的数据
    channel_data->_factor = factor;
    channel_data->_img_h = input_shape_.dims[2] / factor;
    channel_data->_img_w = input_shape_.dims[3] / factor;
    if (channel_data->_net_ouput.data.size() != 0 || channel_data->_net_ouput.shape.size() != 0)
    {
        channel_data->_net_ouput.data.clear();
        channel_data->_net_ouput.shape.clear();
    }
    if (channel_data->need_feature)
    {
        channel_data->_net_ouput.data = net_output[1].data;
        channel_data->_net_ouput.shape = net_output[1].shape;
    }

    result.resize(detections.size());
    for (int i = 0; i < detections.size(); ++i) {
        result[i].x1 = detections[i].x1;
        result[i].y1 = detections[i].y1;
        result[i].x2 = detections[i].x2;
        result[i].y2 = detections[i].y2;
        // cout<<"sign detect: "<<result[i].x1<<", "<<result[i].y1<<", "<<result[i].x2<<", "<<result[i].y2<<endl;
        result[i].score = detections[i].score;
        result[i].class_id = detections[i].class_id;
    }

    /////////////////以上为原版本的yolov5模块的代码//////////////////////
    return true;
};
void softmax(const float* scores, float* probabilities, int size) {
    float max_score = *std::max_element(scores, scores + size);
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        probabilities[i] = std::exp(scores[i] - max_score); // Subtract max for numerical stability
        sum += probabilities[i];
    }

    for (int i = 0; i < size; ++i) {
        probabilities[i] /= sum; // Normalize
    }
}
bool Alg_Module_helmet_detect_in_region::detect_helmet(std::string channel_name, std::shared_ptr<Device_Handle> &handle, std::shared_ptr<QyImage> &input_image, std::vector<Result_item_yolo_sample> &result) 
{
    std::shared_ptr<Channel_data_helmet_detect_in_region> channel_data = std::dynamic_pointer_cast<Channel_data_helmet_detect_in_region>(this->get_channal_data(channel_name));
    std::shared_ptr<Model_cfg_Yolo_helmet_detect_in_region> model_cfg = std::dynamic_pointer_cast<Model_cfg_Yolo_helmet_detect_in_region>(this->get_model_cfg(this->model_name));
    std::shared_ptr<Module_cfg_Yolo_helmet_detect_in_region> module_cfg = std::dynamic_pointer_cast<Module_cfg_Yolo_helmet_detect_in_region>(this->get_module_cfg());

    //检查参数设置
    std::vector<int> classes;
    module_cfg->get_int_vector("classes", classes);
    //获取指定的模型实例
    auto net = this->get_model_instance(this->helmet_model_name);
   // std::cout << "helmet_model_name:" << helmet_model_name << std::endl;
    if (net == nullptr) {
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);       //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        return false;
    }

    //判断模型是否已经加载
    auto input_shapes = net->get_input_shapes();
   // std::cout << "input_shapes.size():" << input_shapes.size() << std::endl;
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;
    }

    auto input_shape_ = input_shapes[0];    //[channel, height, width]

    // 计算图片尺寸(input_image)和模型输入尺寸(input_shape_)的比例
    float factor1 = input_shape_.dims[3] * 1.0 / input_image->get_width();
    float factor2 = input_shape_.dims[2] * 1.0 / input_image->get_height();

    float thres_width;
    float thres_height;
    bool load_param_res = true;
    load_param_res = module_cfg->get_float("check_thres_width", thres_width);
    load_param_res = module_cfg->get_float("check_thres_height", thres_height);

    if(input_image->get_width()<thres_width||input_image->get_height()<thres_height)
    {
        return false;
    }

    float factor = factor1 > factor2 ? factor2 : factor1; // 选择较小的比例
    int target_width = input_image->get_width() * factor;        // 图片需要缩放到的目标宽度
    int target_height = input_image->get_height() * factor;      // 图片需要缩放到的目标高度
    
    if((target_width < 8) || (target_width > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " target_width invalid " << target_width << std::endl;
        return false;
    }
    if((target_height < 8) || (target_height > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " target_height invalid " << target_height << std::endl;
        return false;
    }
    if((input_image->get_width() < 8) || (input_image->get_width() > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.width invalid " << input_image->get_width() << std::endl;
        return false;
    }
    if((input_image->get_height() < 8) || (input_image->get_height() > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.height invalid " << input_image->get_height() << std::endl;
        return false;
    }

    std::vector<Output> net_output;

    std::shared_ptr<QyImage> sub_image = input_image->resize_keep_ratio(input_shape_.dims[3],input_shape_.dims[2],0,QyImage::Padding_mode::Center);
    // sub_image=sub_image->cvtcolor(true);

    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(sub_image);
    net->forward(net_input, net_output);  

    if (net_output.size()>0) {
        
        float* res = (float*)net_output[0].data.data();

        // 第一个任务
        float task1_score[2] = { res[0], res[1] };
        float task1_prob[2];
        softmax(task1_score, task1_prob, 2);
        int task1_class_id = (task1_prob[0] > task1_prob[1]) ? 0 : 1;

        // 第二个任务
        float task2_score[3] = { res[2], res[3], res[4] };
        float task2_prob[3];
        softmax(task2_score, task2_prob, 3);
        int task2_class_id = std::distance(task2_prob, std::max_element(task2_prob, task2_prob + 3));

        // 第三个任务
        float task3_score[3] = { res[5], res[6], res[7] };
        float task3_prob[3];
        softmax(task3_score, task3_prob, 3);
        int task3_class_id = std::distance(task3_prob, std::max_element(task3_prob, task3_prob + 3));


    
        // 人数和头盔数量
        int num_people = task2_class_id + 1; // 人数
        int num_helmets = task3_class_id; // 头盔

        // 判断条件
        if (task1_class_id == 1 && task3_class_id < task2_class_id+1) {
          //  std:cout<<" num_helmets:"<<num_helmets<<endl;
            //std::cout<<"num_people:"<<num_people<<endl;
            // 将任务得分存储在一个向量中
            std::vector<float> scores = {
                task1_prob[task1_class_id],
                task2_prob[task2_class_id],
                task3_prob[task3_class_id]
            };

            // 排序得分
            std::sort(scores.begin(), scores.end());
            Result_item_yolo_sample result1;
            result1.class_id = 1; // 更新class_id为1（电瓶车）
            result1.score = scores[0];
            result.push_back(result1);        
            // 输出结果
     //       std::cout << "Task 1: Class ID = " << task1_class_id << ", Probability = " << task1_prob[task1_class_id] << std::endl;
  //          std::cout << "Task 2: Class ID = " << task2_class_id << ", Probability = " << task2_prob[task2_class_id] << std::endl;
    //        std::cout << "Task 3: Class ID = " << task3_class_id << ", Probability = " << task3_prob[task3_class_id] << std::endl;
            return true;
        }
    }
 



    return false;
};
float calculateIoU(int x1A, int y1A, int x2A, int y2A, int x1B, int y1B, int x2B, int y2B) {
    // 计算交集
    int x1Intersection = std::max(x1A, x1B);
    int y1Intersection = std::max(y1A, y1B);
    int x2Intersection = std::min(x2A, x2B);
    int y2Intersection = std::min(y2A, y2B);

    // 计算交集的宽度和高度
    int intersectionWidth = x2Intersection - x1Intersection;
    int intersectionHeight = y2Intersection - y1Intersection;

    // 如果交集的宽度或高度为负，表示没有交集
    int intersectionArea = (intersectionWidth > 0 && intersectionHeight > 0) 
                           ? intersectionWidth * intersectionHeight 
                           : 0;

    // 计算并集
    int boxAArea = (x2A - x1A) * (y2A - y1A);
    int boxBArea = (x2B - x1B) * (y2B - y1B);
    int unionArea = boxAArea + boxBArea - intersectionArea;

    // 计算 IoU
    return (unionArea > 0) ? (static_cast<float>(intersectionArea) / unionArea) : 0.0f;
}


/*
@param input["image"]
@param output["result"]
*/
bool Alg_Module_helmet_detect_in_region::forward(
    std::string channel_name,
    std::map<std::string, std::shared_ptr<InputOutput>> &input,
    std::map<std::string, std::shared_ptr<InputOutput>> &output)
{
   //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("fount no image in forward.input", this->node_name, Alg_Module_Exception::Stage::inference);
        return false;
    }

    auto model_cfg = std::dynamic_pointer_cast<Model_cfg_Yolo_helmet_detect_in_region>(this->get_model_cfg(this->model_name));
    auto channel_cfg = std::dynamic_pointer_cast<Channel_cfg_helmet_detect_in_region>(this->get_channel_cfg(channel_name));
    auto channel_data = std::dynamic_pointer_cast<Channel_data_helmet_detect_in_region>(this->get_channal_data(channel_name));
    auto module_cfg = std::dynamic_pointer_cast<Module_cfg_Yolo_helmet_detect_in_region>(this->get_module_cfg());

    //获取计算卡推理核心的handle
    std::shared_ptr<Device_Handle> handle;
    if (this->get_device(handle) == false) // 获取计算卡推理核心的handle
    {
        std::cout << "ch " << channel_name << " model " << model_name << "  Error:\t get device failed" << std::endl;
        throw Alg_Module_Exception("get device failed", this->node_name, Alg_Module_Exception::Stage::inference); // 无法获取设备的handle，一般是要么是还未初始化完成，要么是设备驱动异常
        return false;
    }

    std::shared_ptr<QyImage> input_image;

    if(input["image"]->data_type==InputOutput::Type::Image_t){
        input_image=input["image"]->data.image;
        handle = input_image->get_handle();
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;

    }

    //检测非机动车的元素
    std::vector<Result_item_yolo_sample> detections_engineering_work_factor;
    this->detect_engineering_factor(channel_name, handle, input_image, detections_engineering_work_factor);
    // 保存当前检测结果到上一张图片的检测结果
    if (detections_engineering_work_factor.empty()) {
        previous_detections = detections_engineering_work_factor;
    }

    
    // 遍历当前检测结果，查找需要删除的项
    for (auto it = detections_engineering_work_factor.begin(); it != detections_engineering_work_factor.end(); ) {
            bool to_delete = false;
            // 遍历之前的检测结果
            for (const auto &previous_detection : previous_detections) {

                // 计算 IoU
                float iou = calculateIoU(it->x1, it->y1, it->x2, it->y2,
                                          previous_detection.x1, previous_detection.y1,
                                          previous_detection.x2, previous_detection.y2);
               // std::cout << "previous_detection: " << previous_detection.x1 << ", " << previous_detection.y1 << ", " << previous_detection.x2 << ", " << previous_detection.y2 << std::endl;
              //  std::cout << "current_detection: " << it->x1 << ", " << it->y1 << ", " << it->x2 << ", " << it->y2 << std::endl;
              //  std::cout << "iou: " << iou << std::endl;
                if (iou > 0.98) {
                    to_delete = true; // 设置为删除标记
                    break; // 找到一个满足条件的之前结果，退出内层循环
                }
                
            }

            if (to_delete) {
                // 如果满足条件，删除当前事件
                it = detections_engineering_work_factor.erase(it); // 返回下一个迭代器
            } else {
                ++it; // 移动到下一个事件
            }
    }
    previous_detections = detections_engineering_work_factor; // 更新 previous_detections
    //std::cout << "非机动车的数量: " << detections_engineering_work_factor.size() << std::endl;
    // 检测头盔
    std::vector<Result_item_yolo_sample> detections_helmet;  
    

    for (const auto& detection : detections_engineering_work_factor) {
        // 获取检测框的坐标
        float x1 = detection.x1;
        float y1 = detection.y1;
        float x2 = detection.x2;
        float y2 = detection.y2;

        // 确保坐标在有效范围内
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(input_image->get_width())));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(input_image->get_height())));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(input_image->get_width())));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(input_image->get_height())));

        // 创建 cv::Rect 对象
        cv::Rect crop_box(static_cast<int>(x1), static_cast<int>(y1), 
                        static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));

        // 假设原始矩形为 crop_box
        int original_height = crop_box.height;
        int height_add=0;
        if(crop_box.height<crop_box.width*1.2){
            height_add=(crop_box.width)*1.2-crop_box.height;
        }
        else{
            height_add=original_height*0.2;
            height_add=std::min(height_add,input_image->get_height()/15);
        }  
       // height_add=original_height*0.2;
       // height_add=std::min(height_add,input_image->get_height()/15);
        int new_height = original_height +height_add; // 扩大高度为原来的两倍

        // 更新 y 坐标，向上扩大
        int new_y = crop_box.y - height_add; // 向上移动 y 坐标
        if(new_y<0)
            new_y=0;

        // 创建新的 cv::Rect 对象
        cv::Rect expanded_crop_box(crop_box.x, new_y, crop_box.width, new_height);


        // 裁剪图像
        std::shared_ptr<QyImage> sub_image = input_image->crop(expanded_crop_box);

        if (sub_image) {
          /* std::cout << "Sub image dimensions: "
                    << sub_image->get_width() << " x "
                    << sub_image->get_height() << std::endl;*/ 

            // 获取裁剪后的图像宽高
            int width = sub_image->get_width();
            int height = sub_image->get_height();


        }

        

        if (this->detect_helmet(channel_name, handle, sub_image, detections_helmet))
        {
//        std::unordered_map<int, int> class_counts;
        //cout<<"helmet size:"<<detections_helmet.size()<<endl;
            auto& helmet_detection=detections_helmet[detections_helmet.size()-1];

            helmet_detection.x1 = expanded_crop_box.x;
            helmet_detection.y1 = expanded_crop_box.y;
            helmet_detection.x2 = expanded_crop_box.x + expanded_crop_box.width;
            helmet_detection.y2 = expanded_crop_box.y + expanded_crop_box.height;

            }

    }

    auto detect_result = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    detect_result->data.detect.resize(detections_helmet.size());
    auto& results = detect_result->data.detect;

    for (int i = 0; i < detections_helmet.size(); i++) {
        results[i].class_id = detections_helmet[i].class_id;
        results[i].score = detections_helmet[i].score;
        results[i].x1 = range_thred(detections_helmet[i].x1, 0, input_image->get_width());
        results[i].y1 = range_thred(detections_helmet[i].y1, 0, input_image->get_height());
        results[i].x2 = range_thred(detections_helmet[i].x2, 0, input_image->get_width());
        results[i].y2 = range_thred(detections_helmet[i].y2, 0, input_image->get_height());

        
        }
    //std::cout << "helmet detect result size: " << results.size() << std::endl;
    output.clear();
    output["image"] = input["image"];
    output["result"] = detect_result; // 仅在时间条件符合时保存结果  
    return true;
    

};

/*
@param input["result"]
@param output["result"]
*/
bool Alg_Module_helmet_detect_in_region::filter(
    std::string channel_name,
    std::map<std::string, std::shared_ptr<InputOutput>> &input,
    std::map<std::string, std::shared_ptr<InputOutput>> &output)
{
    std::shared_ptr<Channel_data_helmet_detect_in_region> ch_data = std::dynamic_pointer_cast<Channel_data_helmet_detect_in_region>(this->get_channal_data(channel_name));
    if (ch_data == nullptr)
    {
        std::cout << "ch " << channel_name << " model " <<  model_name << "  Error:\t channel instance get fail " << this->node_name << std::endl;
        return false;
    }

    // 将模型输出结果，使用去重、区域验证等方式进行处理，得到事件结果,放到output中

    if (input["result"] == nullptr)
    {
        return false;
    }

    //获取检测结果
    auto detect_res_ = input["result"]->data.detect; // 检测结果



    //没有检测结果
    if (detect_res_.size() <= 0)
    {   
        auto detect_res = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
        detect_res->data.detect.resize(0);
        output.clear();
        output["result"] = detect_res;
        return true;
    } 

    if (ch_data->boundarys_.size() == 0)
    {
       auto detect_res = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
        detect_res->data.detect.resize(0);
        output.clear();
        output["result"] = detect_res;
        std::cout << "ch " << channel_name << " model " <<  model_name << "  不存在边界" << std::endl;
        return false;
    }

    if (ch_data->width == 0 || ch_data->height == 0)
    {
        std::cout << "ch " << channel_name << " model " <<  model_name << "  重置缓冲区" << std::endl;
        ch_data->width = input["image"]->data.image->get_width();
        ch_data->height = input["image"]->data.image->get_height();
        ch_data->init_buffer(ch_data->width, ch_data->height);
    }
    for (auto iter = ch_data->boundarys_.begin(); iter != ch_data->boundarys_.end(); iter++)
    {
        // std::cout << "重置计数结果" << endl;
        int mask_idx = iter->first; // 各个边界在原始图像上的掩码
        ch_data->grid_temp[mask_idx] = cv::Mat::zeros(ch_data->grid_height, ch_data->grid_width, CV_16UC1);

    }

    // 转换到内部处理方法
    vector<Result_item_yolo_sample> detect_res;

//    detect_res.resize(detect_res_.size());
    for (int i = 0; i < detect_res_.size(); i++)
    {   Result_item_yolo_sample result;
        result.class_id = detect_res_[i].class_id;
        result.score = detect_res_[i].score;
        result.x1 = detect_res_[i].x1;
        result.y1 = detect_res_[i].y1;
        result.x2 = detect_res_[i].x2;
        result.y2 = detect_res_[i].y2;
//        if (result.class_id == 1) {
            detect_res.push_back(result);
  //      }
       //cout<<result.str()<<endl;
    }



    ch_data->region_check(detect_res);

    ch_data->count_add();

    ch_data->count_check();

    //先对construction_truck区域进行判定，是否在area       
    std::vector<Result_item_yolo_sample> result;
    //cout<<detect_res.size()<<endl;
    result = ch_data->get_result(detect_res);

    //cout<<result.size()<<endl;
    //ch_data->filter_invalid_objects(result); 
    
    if (result.size() > 0) {
        ch_data->need_remove_duplicate = true;
    } else {
        ch_data->need_remove_duplicate = false;
    }


    if (ch_data->need_remove_duplicate)
    {   
        roi_pooling(ch_data->_net_ouput, result, ch_data->_img_h, ch_data->_img_w);
        //cout<<"去重前result"<<result.size()<<endl;
        result = ch_data->remover.process(result);
        //cout<<"去重后result"<<result.size()<<endl;
    }

    for (int i = 0; i < result.size(); i++)
    {
        result[i].tag = ch_data->decode_tag(result[i]);
        //std::cout << "Tag for element " << i << ": " << result[i].tag << std::endl;
    }
    auto filter_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t); // 结果数据的结构
    auto &filter_results = filter_output->data.detect;                                      // 过滤后的检测结果
    filter_results.resize(result.size());
    for (int i = 0; i < result.size(); i++)
    {
        filter_results[i].class_id = result[i].class_id;
        filter_results[i].score = result[i].score;
        filter_results[i].x1 = result[i].x1;
        filter_results[i].y1 = result[i].y1;
        filter_results[i].x2 = result[i].x2;
        filter_results[i].y2 = result[i].y2;
        filter_results[i].tag = result[i].tag;
        filter_results[i].region_idx = result[i].region_idx;
        filter_results[i].new_obj = result[i].new_obj;
        filter_results[i].temp_idx = result[i].temp_idx;
        filter_results[i].feature = result[i].feature;
        filter_results[i].contour = result[i].contour;
        //cout<<"filter_results[i].tag"<<result[i].tag<<endl;
    }
    output.clear();
    output["result"] = filter_output;

    if (false)
    {
        // 这个阶段的异常为Alg_Module_Exception::Stage::filter
        throw Alg_Module_Exception("some error", this->node_name, Alg_Module_Exception::Stage::filter);
    }

    return true;
};


bool Alg_Module_helmet_detect_in_region::display(
    std::string channel_name,
    std::map<std::string, std::shared_ptr<InputOutput>> &input,
    std::map<std::string, std::shared_ptr<InputOutput>> &output)
{
    if (output["result"] == nullptr)
    {
        return false;
    }

    auto& results = output["result"]->data.detect;
    if (results.size() <= 0) return true;   //没有检测结果

    //加载模块需要的参数
    int box_color_blue;
    int box_color_green;
    int box_color_red;
    module_cfg->get_int("box_color_blue", box_color_blue);
    module_cfg->get_int("box_color_green", box_color_green);
    module_cfg->get_int("box_color_red", box_color_red);

    //获取图片
    cv::Mat image;
    if (input["image"]->data_type == InputOutput::Type::Image_t) {
        image = input["image"]->data.image->get_image();
    }
    else {
        //暂时不支持其他类型的图像
        throw Alg_Module_Exception("Error:\t image input type error",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }
    // 将检查结果绘制到原始图片上
    for (int idx = 0; idx < results.size(); idx++)
    {
        // std::pair<std::string, cv::Mat> res_image1 = {"image", image};
        // results[idx].res_images.insert(res_image1);
      
        int x = results[idx].x1;
        int y = results[idx].y1;
        int w = results[idx].x2 - results[idx].x1;
        int h = results[idx].y2 - results[idx].y1;
        w = std::min(image.cols - x, w);
        h = std::min(image.rows - y, h);


        cv::Mat image_cpy = image.clone();
#if 0  
        std::pair<std::string, cv::Mat> res_image = {"image", image_cpy};
        results[idx].res_images.insert(res_image);

        cv::Mat result_image_cut = image_cpy(cv::Rect(x, y, w, h));
        std::pair<std::string, cv::Mat> res_image2 = {"cut_image", result_image_cut};
        results[idx].res_images.insert(res_image2);
#endif
        cv::Mat result_image = drawBox(image_cpy, results[idx].x1, results[idx].y1, results[idx].x2, results[idx].y2, cv::Scalar(box_color_blue, box_color_green, box_color_red));
        std::pair<std::string, cv::Mat> res_image3 = {"image", result_image};
        results[idx].res_images.insert(res_image3);
    }

    if (false)
    {
        // 这个阶段的异常为Alg_Module_Exception::Stage::filter
        throw Alg_Module_Exception("some error", this->node_name, Alg_Module_Exception::Stage::display);
    }
    // 从input中取图像，filter_output为过滤后的事件的结果
    // 对过滤后的事件结果进行可视化，画框，截取车牌区域(部分算法模块需要)后，放到filter_output需要可视化的结果的结构体的res_images中
    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_helmet_detect_in_region::load_module_cfg_(std::string cfg_path)
{
    // 可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_Yolo_helmet_detect_in_region>(this->node_name);
    res->from_file(cfg_path);
    return res;
};


std::shared_ptr<Model_cfg_base> Alg_Module_helmet_detect_in_region::load_model_cfg_(std::string cfg_path)
{
    // 可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_Yolo_helmet_detect_in_region>();
    res->from_file(cfg_path);
    return res;
};
bool Alg_Module_helmet_detect_in_region::load_channel_cfg(std::string channel_name, std::string cfg_path){
    auto ptr= this->load_channel_cfg_(channel_name,cfg_path);
    if(ptr==nullptr){
        throw Alg_Module_Exception("load channel cfg error from "+cfg_path,this->node_name,Alg_Module_Exception::Stage::load_channel);
        return false;
    }
    this->set_channel_cfg(ptr);
    return true;

};
std::shared_ptr<Channel_cfg_base> Alg_Module_helmet_detect_in_region::load_channel_cfg_(std::string channel_name,std::string cfg_path){
    auto channel_cfg = std::make_shared<Channel_cfg_helmet_detect_in_region>(channel_name);
    int ret = channel_cfg->from_file(cfg_path);
    if (ret < 0)
    {
        channel_cfg.reset();
        std::cout<<"ERROR:\t channel "<<channel_name<<" config file not exists or format error "<<cfg_path<<std::endl;
    }
    else
    {
        //        this->channel_cfg[channel_name]=channel_cfg;
    }
    return channel_cfg;

};







std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> Channel_cfg_helmet_detect_in_region::get_timestamps(std::string name) {

    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> filtered_timestamps;
    if(name==""){
        
         return timestamps;
    }
    // 遍历所有时间戳
    // 打印 timestamps 的内容

    for (const auto& timestamp : timestamps) {
            const std::string& name = timestamp.first;
            const auto& time_pairs = timestamp.second;
            filtered_timestamps.push_back(timestamp);

        }


    return filtered_timestamps; // 返回匹配的时间戳
}

std::shared_ptr<Channel_data_base> Alg_Module_helmet_detect_in_region::init_channal_data_(std::string channel_name)
{
    auto it = channel_datas_.find(channel_name);
    if (it != channel_datas_.end())
    {
        channel_datas_.erase(it);
    }

    auto ch_data = std::make_shared<Channel_data_helmet_detect_in_region>(channel_name);
   
    // 设置boundary
    // 获取通道配置
    auto ch_cfg = get_channel_cfg(channel_name); 
    if (!ch_cfg)
    {
        return {};
    }
    std::shared_ptr<Channel_cfg_helmet_detect_in_region> channel_cfg = std::dynamic_pointer_cast<Channel_cfg_helmet_detect_in_region>(ch_cfg);
    ch_data->set_module_data(this->module_data_);

    std::vector<std::pair<std::string, std::vector<std::pair<float, float>>>>
        boundarys;
    for (auto const &accept_name : module_data_.accepted_boundary_name)
    {   
        if (auto accept_boundary = ch_cfg->get_boundary(accept_name);
            !accept_boundary.empty())
        {
                    // 打印获取到的边界信息
        for (const auto& boundary : accept_boundary)
        {
 
        
            boundarys.insert(boundarys.end(), accept_boundary.begin(),
                             accept_boundary.end());
        }
    }}

    ch_data->set_boundarys(boundarys);
    ch_data->set_need_remove_duplicate(true);
    channel_datas_[channel_name] = ch_data;

    return ch_data;
};

std::shared_ptr<Channel_data_helmet_detect_in_region> Alg_Module_helmet_detect_in_region::get_channel_data(std::string channel_name)
{
    auto it = channel_datas_.find(channel_name);
    if (it != channel_datas_.end())
    {
        return it->second;
    }
    return nullptr;
};

bool Alg_Module_helmet_detect_in_region::reset_channal_data(std::string channel_name)
{
    auto it = channel_datas_.find(channel_name);
    if (it != channel_datas_.end())
    {
        it->second->reset_channal_data();
        return true;
    }

    return false;
}
extern "C" Alg_Module_Base *create() // 外部调用的构造函数
{
    return new Alg_Module_helmet_detect_in_region(); // 返回当前算法模块子类的指针
};

extern "C" void destory(Alg_Module_Base *p) // 外部调用的析构函数
{
    delete p;
};
