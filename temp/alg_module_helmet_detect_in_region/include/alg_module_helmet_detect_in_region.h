#ifndef __ALG_MODULE_helmet_detect_in_region_H__
#define __ALG_MODULE_helmet_detect_in_region_H__
#include "alg_module_base_private.h"
#include "error_type.h"
using namespace std;
#define DEFAULT_FRAME_INTERVAL 12             // 12
#define DEFAULT_TICK_INTERVAL_MS 6000         // 6000
#define DEFAULT_THRESH_IOU 0.4                // 0.4
#define DEFAULT_THRESH_SCORE 0.45             // 0.45
#define DEFAULT_CHECK_SENSITIVITY_THRES 0.5   // 0.5
#define DEFAULT_DUPL_RM_ACCEPT_SIM_THRES 0.7  // 0.7
#define DEFAULT_DUPL_RM_TRIGGER_SIM_THRES 0.0 // 0.0
#define DEFAULT_DUPL_RM_IOU_THRES 0.7         // 0.7

#define DEFAULT_DUPL_RM_MIN_INTERVAL 100 // 100
#define DEFAULT_DUPL_RM_MAX_INTERVAL 500 // 500
#define DEFAULT_BOX_COLOR_RED 255        // 255
#define DEFAULT_BOX_COLOR_GREEN 255      // 255
#define DEFAULT_BOX_COLOR_BLUE 0         // 0

class Result_item_yolo_sample
{ // 模块内使用的YOLO的结果的结构
public:
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float area;
    int class_id;
    std::string tag = "";
    std::vector<float> feature;
    std::vector<std::pair<float, float>> contour;
    int region_idx;
    bool new_obj = false;
    int temp_idx = -1;

    std::string str()
    {
        std::string res;
        res += "class:" + Num2string<int>(this->class_id);
        res += "\tscore:" + Num2string<float>(this->score);
        res += "\tx1:" + Num2string<float>(this->x1);
        res += "\ty1:" + Num2string<float>(this->y1);
        res += "\tx2:" + Num2string<float>(this->x2);
        res += "\ty2:" + Num2string<float>(this->y2);
        res += "\ttag:" + this->tag;
        return res;
    };
};

class Module_data
{
public:
    std::map<int, string> class_id2class_name;
    std::set<string> accepted_boundary_name;
    std::set<string> accepted_time_name;
    std::map<string, std::set<int>> region_depended_class;
    bool check_exist;
    bool need_count_check;
    int check_count_thres;

    float check_sensitivity_thres;
    float check_area_thres;//增加的area_thres
    float dupl_rm_accept_sim_thres;
    float dupl_rm_trigger_sim_thres;
    float dupl_rm_iou_thres;
    int dupl_rm_min_interval;
    int dupl_rm_max_interval;
};

class Duplicate_remover
{
public:
    Duplicate_remover();
    virtual ~Duplicate_remover();
    void set_min_interval(int interval);
    void set_max_interval(int interval);
    void set_accept_sim_thres(float thres);
    void set_trigger_sim_thres(float thres);
    void set_iou_thres(float thres);

    virtual std::vector<Result_item_yolo_sample> process(std::vector<Result_item_yolo_sample> result);

protected:
    std::vector<std::pair<int, float>> check_score(std::vector<Result_item_yolo_sample> &result);
    std::vector<std::pair<Result_item_yolo_sample, int>> last_accept_res;

    float accept_sim_thres;
    float trigger_sim_thres;
    float iou_thres;
    int min_interval;
    int max_interval;
};

class Model_cfg_Yolo_helmet_detect_in_region : public Model_cfg_base
{ // 派生的模型配置文件类
public:
   // bmcv_copy_to_atrr_t copy_to_attr_;

public:
    Model_cfg_Yolo_helmet_detect_in_region()
    {
       /* copy_to_attr_.start_x = 0;
        copy_to_attr_.start_y = 0;
        copy_to_attr_.padding_r = 114;
        copy_to_attr_.padding_g = 114;
        copy_to_attr_.padding_b = 114;
        copy_to_attr_.if_padding = 1;*/
    };
    virtual ~Model_cfg_Yolo_helmet_detect_in_region(){

    };
};

class Module_cfg_Yolo_helmet_detect_in_region : public Module_cfg_base
{ // 派生的模块配置文件类
public:
    Module_cfg_Yolo_helmet_detect_in_region(std::string module_name) : Module_cfg_base(module_name){

                                                                    };

    virtual int from_string_(std::string cfg_str)
    {

        if (this->cfg_int_vector.find("classes") == this->cfg_int_vector.end())
        {
            std::vector<int> classes;
            this->cfg_int_vector["classes"] = classes;
        }

        if (this->cfg_float.find("thresh_iou") == this->cfg_float.end())
        {
            this->cfg_float["thresh_iou"] = DEFAULT_THRESH_IOU;
        }

        if (this->cfg_float.find("thresh_score") == this->cfg_float.end())
        {
            this->cfg_float["thresh_score"] = DEFAULT_THRESH_SCORE;
        }

        if (this->cfg_float.find("check_sensitivity_thres") == this->cfg_float.end())
        {
            this->cfg_float["check_sensitivity_thres"] = DEFAULT_CHECK_SENSITIVITY_THRES;
        }

        if (this->cfg_float.find("dupl_rm_accept_sim_thres") == this->cfg_float.end())
        {
            this->cfg_float["dupl_rm_accept_sim_thres"] = DEFAULT_DUPL_RM_ACCEPT_SIM_THRES;
        }

        if (this->cfg_float.find("dupl_rm_trigger_sim_thres") == this->cfg_float.end())
        {
            this->cfg_float["dupl_rm_trigger_sim_thres"] = DEFAULT_DUPL_RM_TRIGGER_SIM_THRES;
        }

        if (this->cfg_float.find("dupl_rm_iou_thres") == this->cfg_float.end())
        {
            this->cfg_float["dupl_rm_iou_thres"] = DEFAULT_DUPL_RM_IOU_THRES;
        }

        if (this->cfg_int.find("dupl_rm_min_interval") == this->cfg_int.end())
        {
            this->cfg_int["dupl_rm_min_interval"] = DEFAULT_DUPL_RM_MIN_INTERVAL;
        }

        if (this->cfg_int.find("dupl_rm_max_interval") == this->cfg_int.end())
        {
            this->cfg_float["dupl_rm_max_interval"] = DEFAULT_DUPL_RM_MAX_INTERVAL;
        }

        if (this->cfg_int.find("box_color_red") == this->cfg_int.end())
        {
            this->cfg_int["box_color_red"] = DEFAULT_BOX_COLOR_RED;
        }

        if (this->cfg_int.find("box_color_green") == this->cfg_int.end())
        {
            this->cfg_int["box_color_green"] = DEFAULT_BOX_COLOR_GREEN;
        }

        if (this->cfg_int.find("box_color_blue") == this->cfg_int.end())
        {
            this->cfg_int["box_color_blue"] = DEFAULT_BOX_COLOR_BLUE;
        }

        if (this->cfg_int.find("tick_interval") == this->cfg_int.end())
        {
            this->cfg_int["tick_interval"] = DEFAULT_TICK_INTERVAL_MS;
        }

        if (this->cfg_int.find("frame_interval") == this->cfg_int.end())
        {
            this->cfg_int["frame_interval"] = DEFAULT_FRAME_INTERVAL;
        }
        return true;
    };
};

class Channel_data_helmet_detect_in_region : public Channel_data_base
{ // 派生的通道数据类
public:
    Channel_data_helmet_detect_in_region(std::string channel_name)
        : Channel_data_base(channel_name) {}
    virtual ~Channel_data_helmet_detect_in_region(){};
    // 原

    int width = 0;
    int height = 0;
    float grid_factor = 0.2;
    int grid_width = 0;
    int grid_height = 0;

    std::map<int, string> mask_idx2mask_name;
    bool need_feature = false;
    bool need_remove_duplicate = false;
    std::map<int, cv::Mat> grid_temp;
    std::map<int, std::vector<cv::Point2f>> boundarys_;
    std::vector<std::pair<std::string, std::vector<std::pair<string, string>>>> timestamps_;

    std::map<int, cv::Mat> mask_grids;
    std::map<int, int> mask_fg_cnt;
    std::map<int, cv::Mat> count_grids;

    string last_img_timestamp="";
    int last_forword = 0;

    float _factor = 1.0;
    Output _net_ouput;
    float _img_h;
    float _img_w;
    Duplicate_remover remover;

    std::map<int, string> class_id2class_name;
    std::set<string> accepted_boundary_name;

    std::map<string, std::set<int>> region_depended_class;
    bool check_exist;
    bool need_count_check;
    int check_count_thres;
    float check_sensitivity_thres;
    float check_area_thres;
    float dupl_rm_accept_sim_thres;
    float dupl_rm_trigger_sim_thres;
    float dupl_rm_iou_thres;
    int dupl_rm_min_interval;
    int dupl_rm_max_interval;

    void set_module_data(Module_data &module_data);
    void set_class_filter(vector<int> &classes);
    void set_grid_factor(float factor);
    void set_boundarys(std::vector<std::pair<std::string, std::vector<std::pair<float, float>>>> boundarys);
    void set_timestamps(std::vector<std::pair<std::string, std::vector<std::pair<string, string>>>> timestamps);
    void add_boundary(std::string region_name, std::vector<std::pair<float, float>> boundary);
    void set_need_remove_duplicate(bool flag);
    void set_need_features(bool flag);
    void filter_invalid_objects(std::vector<Result_item_yolo_sample> &objects);
    void get_features(vector<Result_item_yolo_sample> &result);
    std::string decode_tag(Result_item_yolo_sample item);
    void region_check(vector<Result_item_yolo_sample> detect_result);
    void count_check();
    void count_add();
    void init_buffer(int width, int height);
    vector<Result_item_yolo_sample> get_result(vector<Result_item_yolo_sample> detect_result);
    vector<Result_item_yolo_sample> find_work_region(vector<Result_item_yolo_sample> &detect_res);
    void reset_channal_data();

    std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> get_boundary(std::string name);
};

class Channel_cfg_helmet_detect_in_region : public Channel_cfg_base
{ // 派生的通道数据类
public:
    Channel_cfg_helmet_detect_in_region(std::string channel_name)
        : Channel_cfg_base(channel_name) {}
    virtual ~Channel_cfg_helmet_detect_in_region(){};
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> get_timestamps(std::string name);
    //int from_string_(std::string cfg_str) ;
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> timestamps;
};

class Alg_Module_helmet_detect_in_region : public Alg_Module_Base_private
{ // 派生的算法模块
public:
    Alg_Module_helmet_detect_in_region();
    virtual ~Alg_Module_helmet_detect_in_region();
    float helmet_thresh_iou;
    float helmet_thresh_score;
    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path); // 加载模块配置文件
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);   // 加载模型配置文件
    virtual std::shared_ptr<Channel_cfg_base> load_channel_cfg_(std::string channel_name,std::string cfg_path);
    bool load_channel_cfg(std::string channel_name, std::string cfg_path);
    virtual std::shared_ptr<Channel_data_base>
    init_channal_data_(std::string channel_name)
        override; // 初始化通道独立的需要记忆的变量，如果继承Channel_data_base子类进行实现，则此函数也需要覆盖
    virtual bool reset_channal_data(std::string channel_name) override;

    std::vector<Result_item_yolo_sample> previous_detections;
    virtual bool init_from_root_dir(std::string root_dir);                                                                                                                   // 读取相关配置并进行初始化
    virtual bool forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output); // 进行模型推理
    virtual bool filter(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output);  // 进行结果过滤
    virtual bool display(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output); // 可视化
    bool detect_engineering_factor(std::string channel_name, std::shared_ptr<Device_Handle> &handle, std::shared_ptr<QyImage> &input_image, std::vector<Result_item_yolo_sample> &result);
    bool detect_helmet(std::string channel_name, std::shared_ptr<Device_Handle> &handle, std::shared_ptr<QyImage> &input_image, std::vector<Result_item_yolo_sample> &result); 
    //  void load_channel(std::string channel_name, std::string cfg_path);

private:
    void init_module_data(std::shared_ptr<Module_cfg_Yolo_helmet_detect_in_region> module_cfg);
    std::shared_ptr<Channel_data_helmet_detect_in_region> get_channel_data(std::string channel_name);

    std::string model_path;
    std::string model_name; // 按模型实际名称进行修改
    std::string model_cfg_path;

        // 行人 检测模型
    std::string helmet_model_path;
    std::string helmet_model_name;
    std::string helmet_model_cfg_path;

    std::map<std::string, std::string> channel_cfgpaths_;
    std::map<std::string, std::shared_ptr<Channel_data_helmet_detect_in_region>> channel_datas_;
    Module_data module_data_;
};

extern "C" Alg_Module_Base *create();        // 外部使用的构造函数的接口声明
extern "C" void destory(Alg_Module_Base *p); // 外部使用的构造函数的接口声明

#endif