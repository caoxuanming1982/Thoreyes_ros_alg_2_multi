#ifndef __ALG_MODULE_LICENSE_PLATE_RECOGNITION_H__
#define __ALG_MODULE_LICENSE_PLATE_RECOGNITION_H__
#include "alg_module_base_private.h"
#include "error_type.h"

class Result_item_License_Plate_Recognition { //模块内使用的YOLO的结果的结构
public:
    float x1;
    float y1;
    float x2;
    float y2;
    
    float landms_x1;
    float landms_y1;
    float landms_x2;
    float landms_y2;
    float landms_x3;
    float landms_y3;
    float landms_x4;
    float landms_y4;

    Result_Detect_license::License_Color license_color;
    Result_Detect_license::License_Type license_type;

    std::string license="";
	int car_idx=-1;
    float score;
    int class_id;
    std::string tag="";
    std::vector<float> feature;

    std::string str() {
        std::string res;
        res+="class:"+Num2string<int>(this->class_id);
        res+="\tscore:"+Num2string<float>(this->score);
        res+="\tx1:"+Num2string<float>(this->x1);
        res+="\ty1:"+Num2string<float>(this->y1);
        res+="\tx2:"+Num2string<float>(this->x2);
        res+="\ty2:"+Num2string<float>(this->y2);
        res+="\ttag:"+this->tag+" "+this->type2str(this->license_type)+" "+this->color2str(this->license_color);

        return res;
    };

    static std::string type2str(Result_Detect_license::License_Type _type) {
        switch (_type) {
            case Result_Detect_license::License_Type::Single: return "Single";
            case Result_Detect_license::License_Type::Double: return "Double";
            default: return "UNKNOWN";
        }
    };

    static std::string color2str(Result_Detect_license::License_Color _color) {
        switch (_color) {
            case Result_Detect_license::License_Color::Blue: return "Blue";
            case Result_Detect_license::License_Color::Green: return "Green";
            case Result_Detect_license::License_Color::Yellow: return "Yellow";
            case Result_Detect_license::License_Color::Yellow_Green: return "Yellow_Green";
            case Result_Detect_license::License_Color::Black: return "Black";
            case Result_Detect_license::License_Color::White: return "White";
            default: return "UNKNOWN";
        }
    };
};

class Model_cfg_License_Plate_Recognition:public Model_cfg_base{ //派生的模型配置文件类
public:
    Model_cfg_License_Plate_Recognition(){
    };
    virtual ~Model_cfg_License_Plate_Recognition(){

    };
};

class Module_cfg_License_Plate_Recognition:public Module_cfg_base{ //派生的模块配置文件类
public:
    Module_cfg_License_Plate_Recognition(std::string module_name):Module_cfg_base(module_name){

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

        //模型推理相关
        if (this->cfg_float.find("single_thresh_score")==this->cfg_float.end()) {
            this->cfg_float["single_thresh_score"] = 0.1;
        }
        if (this->cfg_float.find("double_thresh_score")==this->cfg_float.end()) {
            this->cfg_float["double_thresh_score"] = 0.1;
        }

        return true;
    };
};


class Alg_Module_License_Plate_Recognition:public Alg_Module_Base_private{ //派生的算法模块
public:
    Alg_Module_License_Plate_Recognition();
    virtual ~Alg_Module_License_Plate_Recognition();

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);    //加载模块配置文件
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);      //加载模型配置文件

    virtual bool init_from_root_dir(std::string root_dir);                              //读取相关配置并进行初始化
    virtual bool forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& output);    //进行模型推理
    virtual bool filter(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& output);     //进行结果过滤
    virtual bool display(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input,std::map<std::string,std::shared_ptr<InputOutput>>& filter_output);    //可视化

    virtual bool recognize_license_plate(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::vector<Result_Detect> &vehicles, std::vector<Result_Detect_license> &licenses, std::vector<Result_item_License_Plate_Recognition> &licenses_res);
    std::string recognize_single_license(std::string channel_name, std::shared_ptr<QyImage>& input_image, Result_Detect vehicle, Result_Detect_license license);
    std::string recognize_double_license(std::string channel_name, std::shared_ptr<QyImage>& input_image, Result_Detect vehicle, Result_Detect_license license);

    std::shared_ptr<QyImage> get_sub_image_with_perspect(std::string channel_name, std::shared_ptr<QyImage>& input,Result_Detect& vehicle, Result_Detect_license& license,int w=0,int h=0);
    std::shared_ptr<QyImage> get_image(std::map<std::string, std::shared_ptr<InputOutput>>& input);


    // 单行车牌的识别模型
    std::string single_model_name;
    std::string single_model_path;
    std::string single_model_cfg_path;

    // 双行车牌的识别模型
    std::string model_name;
    std::string model_path;
    std::string model_cfg_path;
};

extern "C" Alg_Module_Base* create();           //外部使用的构造函数的接口声明
extern "C" void destory(Alg_Module_Base* p);    //外部使用的构造函数的接口声明

#endif