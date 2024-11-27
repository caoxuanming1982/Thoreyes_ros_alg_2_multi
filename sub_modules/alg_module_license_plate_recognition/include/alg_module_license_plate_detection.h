#ifndef __ALG_MODULE_LICENSE_PLATE_DETECTION_H__
#define __ALG_MODULE_LICENSE_PLATE_DETECTION_H__
#include "alg_module_base_private.h"
#include "error_type.h"

class Result_item_License_Plate_Detection {
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

class Model_cfg_License_Plate_Detection:public Model_cfg_base{
public:
    Model_cfg_License_Plate_Detection(){
    };
    virtual ~Model_cfg_License_Plate_Detection(){

    };
};

class Module_cfg_License_Plate_Detection:public Module_cfg_base{
public:
    Module_cfg_License_Plate_Detection(std::string module_name):Module_cfg_base(module_name){

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
        if (this->cfg_float.find("min_lp_width")==this->cfg_float.end()) {
            this->cfg_float["min_lp_width"] = 20;
        }
        if (this->cfg_float.find("min_lp_height")==this->cfg_float.end()) {
            this->cfg_float["min_lp_height"] = 10;
        }
        if (this->cfg_float.find("thresh_iou")==this->cfg_float.end()) {
            this->cfg_float["thresh_iou"] = 0.5;
        }
        if (this->cfg_float.find("thresh_score")==this->cfg_float.end()) {
            this->cfg_float["thresh_score"] = 0.5;
        }
        if (this->cfg_int_vector.find("classes")==this->cfg_int_vector.end()) {
            std::vector<int> classes;
            this->cfg_int_vector["classes"] = classes;
        }
        return true;
    };
};

class Alg_Module_License_Plate_Detection:public Alg_Module_Base_private{
public:
    Alg_Module_License_Plate_Detection();
    virtual ~Alg_Module_License_Plate_Detection();

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);

//    virtual bool detect_license_plate(string channel_name, bm_handle_t &handle, bm_image &input_image, std::vector<Result_Detect> &vehicles, std::vector<Result_item_License_Plate_Detection> &licenses);
    virtual bool detect_license_plate(string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::vector<Result_Detect> &vehicles, std::vector<Result_item_License_Plate_Detection> &licenses);

    virtual bool init_from_root_dir(std::string root_dir);
    virtual bool forward(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& output);
    virtual bool filter(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& output);
    virtual bool display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& filter_output);

    std::string model_path;
    std::string model_name;
    std::string model_cfg_path;
};

extern "C" Alg_Module_Base* create();
extern "C" void destory(Alg_Module_Base* p);

#endif