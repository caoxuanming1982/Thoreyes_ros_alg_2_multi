#ifndef __ALG_MODULE_BURST_INTO_BAN_DETECTION_H__
#define __ALG_MODULE_BURST_INTO_BAN_DETECTION_H__
#include "alg_module_base_private.h"
#include "error_type.h"
#ifdef HAVE_OMP
#include <omp.h>
#endif

class Result_item_burst_into_ban_detection {    //模块内使用的YOLO的结果的结构
public:
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
    int region_idx;
    int temp_idx=-1;
    bool new_obj=false;
    std::string tag="";
    std::vector<float> feature;
    std::vector<std::pair<float,float>> contour;
    float worker_conf;
    std::string str(){
        std::string res;
        res+="class:"+Num2string<int>(this->class_id);
        res+="\tscore:"+Num2string<float>(this->score);
        res+="\tx1:"+Num2string<float>(this->x1);
        res+="\ty1:"+Num2string<float>(this->y1);
        res+="\tx2:"+Num2string<float>(this->x2);
        res+="\ty2:"+Num2string<float>(this->y2);
        res+="\ttag:"+this->tag;
        return res;
    };
};

class Duplicate_remover {
public:
    Duplicate_remover();
    virtual ~Duplicate_remover();
    void set_min_interval(int interval);
    void set_max_interval(int interval);
    void set_accept_sim_thres(float thres);
    void set_trigger_sim_thres(float thres);
    void set_iou_thres(float thres);

    virtual std::vector<Result_item_burst_into_ban_detection> process(std::vector<Result_item_burst_into_ban_detection> result);
    
protected:
    std::vector<std::pair<int,float>> check_score(std::vector<Result_item_burst_into_ban_detection> &result);
    std::vector<std::pair<Result_item_burst_into_ban_detection,int>> last_accept_res;
    
    //需要在配置文件中设置的参数
    float accept_sim_thres;
    float trigger_sim_thres;
    float iou_thres;
    int min_interval;
    int max_interval;
};

class Model_cfg_Burst_Into_Ban_Detection:public Model_cfg_base {    //派生的模型配置文件类
public:
    Model_cfg_Burst_Into_Ban_Detection(){
    };
    virtual ~Model_cfg_Burst_Into_Ban_Detection(){

    };
};

class Module_cfg_Burst_Into_Ban_Detection:public Module_cfg_base {  //派生的模块配置文件类
public:
    Module_cfg_Burst_Into_Ban_Detection(std::string module_name):Module_cfg_base(module_name) {

    };

    virtual int from_string_(std::string cfg_str) {
        //目标框相关
        if (this->cfg_int.find("box_color_blue")==this->cfg_int.end()) {
            this->cfg_int["box_color_blue"] = 255;
        }
        if (this->cfg_int.find("box_color_green")==this->cfg_int.end()) {
            this->cfg_int["box_color_green"] = 0;
        }
        if (this->cfg_int.find("box_color_red")==this->cfg_int.end()) {
            this->cfg_int["box_color_red"] = 0;
        }
        if (this->cfg_int.find("box_thickness")==this->cfg_int.end()) {
            this->cfg_int["box_thickness"] = 3;
        }

        //Duplicate_remover_spoil相关
        if (this->cfg_int.find("min_interval")==this->cfg_int.end()) {
            this->cfg_int["min_interval"] = 100;
        }
        if (this->cfg_int.find("max_interval")==this->cfg_int.end()) {
            this->cfg_int["max_interval"] = 500;
        }
        if (this->cfg_float.find("accept_sim_thres")==this->cfg_float.end()) {
            this->cfg_float["accept_sim_thres"] = 0.7;
        }
        if (this->cfg_float.find("trigger_sim_thres")==this->cfg_float.end()) {
            this->cfg_float["trigger_sim_thres"] = 0.0;
        }
        if (this->cfg_float.find("iou_thres")==this->cfg_float.end()) {
            this->cfg_float["iou_thres"] = 0.7;
        }

        //Channel_data相关
        if (this->cfg_float.find("man_non_motor_thresh_iou")==this->cfg_float.end()) {
            this->cfg_float["man_non_motor_thresh_iou"] = 0.5;
        }
        if (this->cfg_float.find("man_thresh_score")==this->cfg_float.end()) {
            this->cfg_float["man_thresh_score"] = 0.1;
        }
        if (this->cfg_float.find("non_motor_thresh_score")==this->cfg_float.end()) {
            this->cfg_float["non_motor_thresh_score"] = 0.1;
        }
        if (this->cfg_float.find("check_sensitivity_thres")==this->cfg_float.end()) {
            this->cfg_float["check_sensitivity_thres"] = 0.5;
        }
        
        //模型推理相关
        if (this->cfg_float.find("thresh_iou") == this->cfg_float.end()) {
            this->cfg_float["thresh_iou"] = 0.4;
        }
        if (this->cfg_float.find("thresh_score") == this->cfg_float.end()) { 
            this->cfg_float["thresh_score"] = 0.45;
        }
        if (this->cfg_int_vector.find("classes") == this->cfg_int_vector.end()) { 
            std::vector<int> classes;
            this->cfg_int_vector["classes"] = classes;
        }

        if (this->cfg_float.find("engineering_worker_score") == this->cfg_float.end()) { 
            this->cfg_float["engineering_worker_score"] = 0.5;
        }
        if (this->cfg_float.find("engineering_worker_thresh") == this->cfg_float.end()) { 
            this->cfg_float["engineering_worker_thresh"] = 0.0855;
        }

        return true;
    };
    
};

class Channel_cfg_Burst_Into_Ban_Detection:public Channel_cfg_base {
public:
    Channel_cfg_Burst_Into_Ban_Detection(std::string channel_name);
    virtual ~Channel_cfg_Burst_Into_Ban_Detection();

    std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> copy_bounary();
};

class Channel_data_Burst_Into_Ban_Detection:public Channel_data_base {
public:
    Channel_data_Burst_Into_Ban_Detection(std::string channel_name);
    virtual ~Channel_data_Burst_Into_Ban_Detection();
    
    //需要在配置文件中设置的参数
    float man_non_motor_thresh_iou;
    float man_thresh_score;
    float non_motor_thresh_score;
    float check_sensitivity_thres;

    std::set<string> accepted_boundary_name;
    std::map<int,string> class_id2class_name;
    std::map<int,string> mask_idx2mask_name;
    std::map<string,std::set<int>> region_depended_class;
    std::map<int,std::vector<cv::Point2f>> boundarys_;
    std::map<int,cv::Mat> grid_temp;
    std::map<int,cv::Mat> mask_grids;
    std::map<int,int> mask_fg_cnt;
    std::map<int,cv::Mat> count_grids;
    int check_count_thres = 1;
    bool need_count_check = true;
    bool check_exist = true;
    bool need_remove_duplicate = true;
    bool need_feature = true;

    float factor = 1.0;
    float img_h;
    float img_w;
    int width = 0;
    int height = 0;
    float grid_factor = 0.1;
    int grid_width = 0;
    int grid_height = 0;
    Output net_ouput;
    std::vector<Result_item_burst_into_ban_detection> result_orig;
    Duplicate_remover remover;

    void init_buffer(int width,int height);
    void set_boundarys(std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> boundary);
    void set_class_filter(std::vector<int> &classes);
    void set_thresh_iou(float thres);
    void set_thresh_score(float score);
    void set_check_count_thres(int thres);
    void set_grid_factor(float factor);
    void set_check_sensitivity_thres(float thres);
    void add_boundary(std::string region_name,std::vector<std::pair<float,float>> boundary);
    void set_need_remove_duplicate(bool flag);
    void set_need_features(bool flag);
    void get_features(std::vector<Result_item_burst_into_ban_detection> &result);
    std::string decode_tag(Result_item_burst_into_ban_detection item);
    void region_check(std::vector<Result_item_burst_into_ban_detection> detect_result);
    void count_check();
    void count_add();

    std::vector<Result_item_burst_into_ban_detection> remove_person_in_car(std::vector<Result_item_burst_into_ban_detection> yolo_res, float iou_threshold);
    float bboxSimilarity(const Result_item_burst_into_ban_detection& a, const Result_item_burst_into_ban_detection& b);

    std::vector<Result_item_burst_into_ban_detection> get_result(std::vector<Result_item_burst_into_ban_detection> detect_result);
};

class Alg_Module_Burst_Into_Ban_Detection:public Alg_Module_Base_private { //派生的算法模块
public:
    Alg_Module_Burst_Into_Ban_Detection();
    virtual ~Alg_Module_Burst_Into_Ban_Detection();

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);    //加载模块配置文件
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);      //加载模型配置文件
    virtual std::shared_ptr<Channel_cfg_base> load_channel_cfg_(std::string channel_name,std::string cfg_path); //加载通道配置文件
    virtual std::shared_ptr<Channel_data_base> init_channal_data_(std::string channel_name);    //初始化通道独立的需要记忆的变量

    virtual bool init_from_root_dir(std::string root_dir);                              //读取相关配置并进行初始化
    virtual bool forward(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output);    //进行模型推理
    virtual bool filter(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output);     //进行结果过滤
    virtual bool display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &filter_output);    //可视化

    virtual bool detect_person(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::vector<Result_item_burst_into_ban_detection> &detections);           //进行模型推理

    bool is_police_forward(std::map<std::string,std::shared_ptr<InputOutput>> &input, int x1, int y1, int x2, int y2);    //进行模型推理
    bool classify_engineering_worker(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>> &input, std::vector<Result_item_burst_into_ban_detection> &results);    //进行模型推理
    int debug =0;

private:    
    std::string model_path;
    std::string model_name;
    std::string model_cfg_path;

    std::string is_police_model_name;
    std::string is_police_model_path;
    std::string is_police_model_cfg_path;

    std::string worker_classify_model_name;
    std::string worker_classify_model_path;
    std::string worker_classify_model_cfg_path;

    int debug_count = 0;
    float remove_person_iou_thres=0.9;
};

extern "C" Alg_Module_Base* create();               //外部使用的构造函数的接口声明
extern "C" void destory(Alg_Module_Base* p);        //外部使用的构造函数的接口声明

#endif