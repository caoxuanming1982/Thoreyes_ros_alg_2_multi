#include "alg_module_license_plate_detection.h"
#include <iostream>


float iou(Result_item_License_Plate_Detection &box1, Result_item_License_Plate_Detection &box2)
{
    float x1 = std::max(box1.x1, box2.x1);      //left
    float y1 = std::max(box1.y1, box2.y1);      //top
    float x2 = std::min((box1.x2), (box2.x2));  //right
    float y2 = std::min((box1.y2), (box2.y2));  //bottom
    if(x1>=x2||y1>=y2)
        return 0;
    float over_area = (x2 - x1) * (y2 - y1);
    float box1_w = box1.x2 - box1.x1;
    float box1_h = box1.y2 - box1.y1;
    float box2_w = box2.x2 - box2.x1;
    float box2_h = box2.y2 - box2.y1;
    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);
    return iou;
};
bool sort_score(Result_item_License_Plate_Detection &box1, Result_item_License_Plate_Detection &box2)
{
    return (box1.score > box2.score);


};
inline float fast_exp(float x) 
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
};
inline float sigmoid(float x) 
{
    if (x >= 87) return 1;
    if (x <= -87) return 0;
    return 1.0f / (1.0f + fast_exp(-x));
    // return 1.0f / (1.0f + exp(-x));
};
inline int find_max(float* arr, int length)
{
    int max_idx = 0;
    float max = arr[max_idx];

    for (int idx = 1; idx < length; idx++)
    {
        if (max < arr[idx])
        {
            max_idx = idx;
            max = arr[idx];
        }
    }
    return max_idx;
};
inline Result_Detect_license::License_Type scale_license_plate_type(float* _type, int length=3)
{

    int _t = find_max(_type, length);

    switch (_t) {
        case 0: return Result_Detect_license::License_Type::Single;
        case 1: return Result_Detect_license::License_Type::Double;
        default: return Result_Detect_license::License_Type::Type_UNKNOWN;
    }

};
inline Result_Detect_license::License_Color scale_license_plate_color(float* _color, int length=7)
{
    int _c = find_max(_color, length);

    switch (_c) {
        case 0: return Result_Detect_license::License_Color::Blue;
        case 1: return Result_Detect_license::License_Color::Green;
        case 2: return Result_Detect_license::License_Color::Yellow;
        case 3: return Result_Detect_license::License_Color::Yellow_Green;
        case 4: return Result_Detect_license::License_Color::Black;
        case 5: return Result_Detect_license::License_Color::White;
        default: return Result_Detect_license::License_Color::Color_UNKNOWN;
    }
};
void nms_yolo_v1(std::vector<Output> &net_output, std::vector<Result_item_License_Plate_Detection> &output, std::vector<int> class_filter,float threshold_score=0.5,float threshold_iou=0.5) 
{

    // net_output 特征图
    // net_output[0] 1,3,40,40,24 ✔
    // net_output[1] 1,3,20,20,24 ✔
    // net_output[2] 1,3,10,10,24 ✔
    
    int stride[3] = {8, 16, 32};
    int anchors[3][3][2] = {
        {{4, 5}, {8, 10}, {13, 16}},
        {{23, 29}, {43, 55}, {73, 105}},
        {{146, 217}, {231, 300}, {335, 433}},
    };

    // dim4: x1, y1, x2, y2, conf, landms8, type3, color7, cls

    std::vector<Result_item_License_Plate_Detection> result;

    for (int channel = 0; channel < net_output.size(); channel++)
    {
        float* y = (float*)net_output[channel].data.data();
        int dim1 = net_output[channel].shape[1]; // 3
        int dim2 = net_output[channel].shape[2]; // 40/20/10
        int dim3 = net_output[channel].shape[3]; // 40/20/10
        int dim4 = net_output[channel].shape[4]; // 24
        
        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {   
                    int idx = d1*dim2*dim3*dim4 + d2*dim3*dim4 + d3*dim4;

                    // 先计算置信度筛选目标, 后计算具体结果
                    float obj_conf = sigmoid(y[idx+4]); //conf
                    float obj_cls = sigmoid(y[idx+23]); //cls
                    float score = obj_conf * obj_cls;

                    if (score > threshold_score) 
                    {
                        //这里是原版yolo
                        float box_x = sigmoid(y[idx+0]);
                        float box_y = sigmoid(y[idx+1]);
                        float box_w = sigmoid(y[idx+2]);
                        float box_h = sigmoid(y[idx+3]);
                        // float obj_conf = sigmoid(y[idx+4]); //conf
                        // float obj_cls = sigmoid(y[idx+23]); //cls
                        // float score = obj_conf * obj_cls;

                        box_x = (box_x * 2 - 0.5 + d3) * stride[channel]; //x
                        box_y = (box_y * 2 - 0.5 + d2) * stride[channel]; //y
                        box_w = std::pow(box_w * 2, 2) * anchors[channel][d1][0]; //w
                        box_h = std::pow(box_h * 2, 2) * anchors[channel][d1][1]; //h

                        //这里是添加的车牌类型、车牌颜色和车牌角点的结果
                        float type_single   = sigmoid(y[idx+13]); //type: single
                        float type_double   = sigmoid(y[idx+14]); //type: double 
                        float type_unknown  = sigmoid(y[idx+15]); //type: none 
                        float color_blue    = sigmoid(y[idx+16]); //color: blue
                        float color_green   = sigmoid(y[idx+17]); //color: green
                        float color_yellow  = sigmoid(y[idx+18]); //color: yellow
                        float color_yellow_green = sigmoid(y[idx+19]); //color: yellow_green
                        float color_black   = sigmoid(y[idx+20]); //color: black
                        float color_white   = sigmoid(y[idx+21]); //color: white
                        float color_unknown = sigmoid(y[idx+22]); //color: none

                        float landms_x1 = y[idx+5] * anchors[channel][d1][0] + d3 * stride[channel]; //x1
                        float landms_y1 = y[idx+6] * anchors[channel][d1][1] + d2 * stride[channel]; //y1
                        float landms_x2 = y[idx+7] * anchors[channel][d1][0] + d3 * stride[channel]; //x2
                        float landms_y2 = y[idx+8] * anchors[channel][d1][1] + d2 * stride[channel]; //y2
                        float landms_x3 = y[idx+9] * anchors[channel][d1][0] + d3 * stride[channel]; //x3
                        float landms_y3 = y[idx+10] * anchors[channel][d1][1] + d2 * stride[channel]; //y3
                        float landms_x4 = y[idx+11] * anchors[channel][d1][0] + d3 * stride[channel]; //x4
                        float landms_y4 = y[idx+12] * anchors[channel][d1][1] + d2 * stride[channel]; //y4

                        // break Alg_Module_License_Plate_Detection.cpp:131
                        Result_item_License_Plate_Detection item;

                        item.x1 = box_x - box_w/2;
                        item.y1 = box_y - box_h/2;
                        item.x2 = box_x + box_w/2;
                        item.y2 = box_y + box_h/2;

                        item.landms_x1 = landms_x1;
                        item.landms_y1 = landms_y1;
                        item.landms_x2 = landms_x2;
                        item.landms_y2 = landms_y2;
                        item.landms_x3 = landms_x3;
                        item.landms_y3 = landms_y3;
                        item.landms_x4 = landms_x4;
                        item.landms_y4 = landms_y4;

                        float _type[3] = {type_single, type_double, type_unknown};
                        float _color[7] = {color_blue, color_green, color_yellow, color_yellow_green, color_black, color_white, color_unknown};
                        item.license_type = scale_license_plate_type(_type);
                        item.license_color = scale_license_plate_color(_color);

                        item.score = score;
                        result.push_back(item);
                    }
                }
            }
        }
    }

    output.clear();
    if (result.size() <= 0)
        return;

    // nms
    std::sort(result.begin(), result.end(), sort_score);
    output.push_back(result[0]);    //取分数最高的即可
    // std::cout << output[0].str() << std::endl;
    // for (int idx = 0; idx < 3 && idx < result.size(); idx++)
    // {
    //     output.push_back(result[idx]);
    //     std::cout << result[idx].str() << std::endl;
    // }
    
    return ;
};
void nms_yolo(Output &net_output, std::vector<Result_item_License_Plate_Detection> &output, std::vector<int> class_filter,float threshold_score=0.5,float threshold_iou=0.5) 
{
    float *input=(float*)net_output.data.data();
    int dim1=net_output.shape[1];
    int dim2=net_output.shape[2];

    float best_score = 0;
    int result_index = 0;

    // 找到分数最高的车牌目标
    for (int k = 0, i = 0; k < dim1; k++, i += dim2) {
        float score = sigmoid(input[i+9]); //conf
        if (score > threshold_score && score > best_score) {
            result_index = i;
            best_score = score;
        }
    }

    // 完成后续计算
    Result_item_License_Plate_Detection best_result;
    best_result.score = best_score;
    int i = result_index;

    float box_x = (sigmoid(input[i+5])*2 + input[i])   * input[i+4];
    float box_y = (sigmoid(input[i+6])*2 + input[i+1]) * input[i+4];
    float box_w = std::pow(sigmoid(input[i+7])*2, 2)   * input[i+2];
    float box_h = std::pow(sigmoid(input[i+8])*2, 2)   * input[i+3];


    //这里是添加的车牌类型、车牌颜色和车牌角点的结果
    float type_single   = sigmoid(input[i+18]); //type: single
    float type_double   = sigmoid(input[i+19]); //type: double 
    float type_unknown  = sigmoid(input[i+20]); //type: none 
    float color_blue    = sigmoid(input[i+21]); //color: blue
    float color_green   = sigmoid(input[i+22]); //color: green
    float color_yellow  = sigmoid(input[i+23]); //color: yellow
    float color_yellow_green = sigmoid(input[i+24]); //color: yellow_green
    float color_black   = sigmoid(input[i+25]); //color: black
    float color_white   = sigmoid(input[i+26]); //color: white
    float color_unknown = sigmoid(input[i+27]); //color: none
    
    float landms_x1 = input[i+10] * input[i+2] + (input[i]+0.5)   * input[i+4]; //x1
    float landms_y1 = input[i+11] * input[i+3] + (input[i+1]+0.5) * input[i+4]; //y1
    float landms_x2 = input[i+12] * input[i+2] + (input[i]+0.5)   * input[i+4]; //x2
    float landms_y2 = input[i+13] * input[i+3] + (input[i+1]+0.5) * input[i+4]; //y2
    float landms_x3 = input[i+14] * input[i+2] + (input[i]+0.5)   * input[i+4]; //x3
    float landms_y3 = input[i+15] * input[i+3] + (input[i+1]+0.5) * input[i+4]; //y3
    float landms_x4 = input[i+16] * input[i+2] + (input[i]+0.5)   * input[i+4]; //x4
    float landms_y4 = input[i+17] * input[i+3] + (input[i+1]+0.5) * input[i+4]; //y4

    best_result.x1 = box_x - box_w/2;
    best_result.y1 = box_y - box_h/2;
    best_result.x2 = box_x + box_w/2;
    best_result.y2 = box_y + box_h/2;

    best_result.landms_x1 = landms_x1;
    best_result.landms_y1 = landms_y1;
    best_result.landms_x2 = landms_x2;
    best_result.landms_y2 = landms_y2;
    best_result.landms_x3 = landms_x3;
    best_result.landms_y3 = landms_y3;
    best_result.landms_x4 = landms_x4;
    best_result.landms_y4 = landms_y4;

    float _type[3] = {type_single, type_double, type_unknown};
    float _color[7] = {color_blue, color_green, color_yellow, color_yellow_green, color_black, color_white, color_unknown};
    best_result.license_type = scale_license_plate_type(_type);
    best_result.license_color = scale_license_plate_color(_color);


    output.clear();
    if (best_result.score > threshold_score) {
        output.push_back(best_result); //取分数最高的即可
    } 
    return ;
};

Alg_Module_License_Plate_Detection::Alg_Module_License_Plate_Detection():Alg_Module_Base_private("license_plate_detection")
{   //参数是模块名，使用默认模块名初始化

};
Alg_Module_License_Plate_Detection::~Alg_Module_License_Plate_Detection()
{

};


bool Alg_Module_License_Plate_Detection::detect_license_plate(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::vector<Result_Detect> &vehicles, std::vector<Result_item_License_Plate_Detection> &licenses)
{
    std::shared_ptr<Model_cfg_License_Plate_Detection> model_cfg = std::dynamic_pointer_cast<Model_cfg_License_Plate_Detection>(this->get_model_cfg(this->model_name));
    std::shared_ptr<Module_cfg_License_Plate_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Detection>(this->get_module_cfg());

    std::shared_ptr<Device_Handle> handle;
    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;

    }


    // 获取指定的模型实例
    auto net = this->get_model_instance(this->model_name);

    if (net == nullptr) { 
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);   //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        return false;
    }

    auto input_shapes = net->get_input_shapes();
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference); 
        return false;   
    }

    std::vector<int> classes;
    float thresh_iou;
    float thresh_score;
    float min_lp_width;
    float min_lp_height;

    float min_car_width;
    float min_car_height;

    bool load_res = true;
    load_res &= module_cfg->get_float("thresh_score", thresh_score);
    load_res &= module_cfg->get_float("thresh_iou", thresh_iou);
    load_res &= module_cfg->get_float("min_lp_width", min_lp_width);
    load_res &= module_cfg->get_float("min_lp_height", min_lp_height);

    load_res &= module_cfg->get_float("min_car_width", min_car_width);
    load_res &= module_cfg->get_float("min_car_height", min_car_height);

    if (load_res == false) {
        throw Alg_Module_Exception("Error:\t load module param failed",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }

    int width=input_image->get_width();
    int height=input_image->get_height();


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (auto &vehicle : vehicles)
    {

        if (vehicle.x1 < 0) vehicle.x1 = 0;
        if (vehicle.y1 < 0) vehicle.y1 = 0;
        if (vehicle.x2 >= width) vehicle.x2 = width-1;
        if (vehicle.y2 >= height) vehicle.y2 = height-1;

        cv::Rect crop_rect;
        crop_rect.x = vehicle.x1;
        crop_rect.y = vehicle.y1;
        crop_rect.width = vehicle.x2 - vehicle.x1;
        crop_rect.height= vehicle.y2 - vehicle.y1;

        float image_width = vehicle.x2 - vehicle.x1;
        float image_height = vehicle.y2 - vehicle.y1;

        if(image_width<min_car_width || image_height<min_car_height)
            continue;

        if((image_width < 8) || (image_width > 8192)){
            std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.width invalid " << width << std::endl;
            return false;
        }
        if((image_height < 8) || (image_height > 8192)){
            std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.height invalid " << height << std::endl;
            return false;
        }

        /////////////////以下为原版本的yolov5模块的代码//////////////////////
        auto input_shape_ = input_shapes[0];    //模型输入尺寸: [channel, height, width]

        //计算图片尺寸(input_image)和模型输入尺寸(input_shape_)的比例
        float factor1 = input_shape_.dims[3] * 1.0 / image_width;
        float factor2 = input_shape_.dims[2] * 1.0 / image_height;


        float factor = factor1 > factor2 ? factor2 : factor1;       //选择较小的比例
        int target_width = image_width * factor;                    //图片需要缩放到的目标宽度
        int target_height = image_height * factor;                  //图片需要缩放到的目标高度
        std::vector<Output> net_output;

        std::vector<std::shared_ptr<QyImage>> net_input;
        std::shared_ptr<QyImage> sub_image=input_image->crop_resize_keep_ratio(crop_rect,input_shape_.dims[3],input_shape_.dims[2],0);
        sub_image=sub_image->cvtcolor(true);
        net_input.push_back(sub_image);
        net->forward(net_input, net_output);  


        std::vector<Result_item_License_Plate_Detection> detections;
        // nms_yolo(net_output, detections, classes, thresh_score, thresh_iou);    // v1.0

        nms_yolo(net_output[0], detections, classes, thresh_score, thresh_iou); // v2.0

        /////////////////以上为原版本的yolov5模块的代码//////////////////////

        if (detections.size() <= 0) continue;
        
        float dw = image_width;   // 原始图片宽度
        float dh = image_height;  // 原始图片高度
        float dw_div = target_width/dw; // 缩放后图片的宽度
        float dh_div = target_height/dh;// 缩放后图片的高度

        for (int i = 0; i < detections.size(); i++) {   
            //从320x320映射回原始图像上
            detections[i].landms_x1 = detections[i].landms_x1/dw_div;
            detections[i].landms_x2 = detections[i].landms_x2/dw_div;
            detections[i].landms_x3 = detections[i].landms_x3/dw_div;
            detections[i].landms_x4 = detections[i].landms_x4/dw_div;
            detections[i].landms_y1 = detections[i].landms_y1/dh_div;
            detections[i].landms_y2 = detections[i].landms_y2/dh_div;
            detections[i].landms_y3 = detections[i].landms_y3/dh_div;
            detections[i].landms_y4 = detections[i].landms_y4/dh_div;

            //避免目标框超出图片范围
            if (detections[i].landms_x1 < 0) detections[i].landms_x1 = 0;
            if (detections[i].landms_x2 < 0) detections[i].landms_x2 = 0;
            if (detections[i].landms_x3 < 0) detections[i].landms_x3 = 0;
            if (detections[i].landms_x4 < 0) detections[i].landms_x4 = 0;
            if (detections[i].landms_y1 < 0) detections[i].landms_y1 = 0;
            if (detections[i].landms_y2 < 0) detections[i].landms_y2 = 0;
            if (detections[i].landms_y3 < 0) detections[i].landms_y3 = 0;
            if (detections[i].landms_y4 < 0) detections[i].landms_y4 = 0;
            
            if (detections[i].landms_x1 > image_width) detections[i].landms_x1 = image_width;
            if (detections[i].landms_x2 > image_width) detections[i].landms_x2 = image_width;
            if (detections[i].landms_x3 > image_width) detections[i].landms_x3 = image_width;
            if (detections[i].landms_x4 > image_width) detections[i].landms_x4 = image_width;
            if (detections[i].landms_y1 > image_height) detections[i].landms_y1 = image_height;
            if (detections[i].landms_y2 > image_height) detections[i].landms_y2 = image_height;
            if (detections[i].landms_y3 > image_height) detections[i].landms_y3 = image_height;
            if (detections[i].landms_y4 > image_height) detections[i].landms_y4 = image_height;

            // 过滤掉过小的车牌
            int left  = detections[i].landms_x1;
            int top   = detections[i].landms_y1;
            int right = detections[i].landms_x3;
            int bot   = detections[i].landms_y3;
            if (left  > detections[i].landms_x4) left  = detections[i].landms_x4;
            if (top   > detections[i].landms_y2) top   = detections[i].landms_y2;
            if (right < detections[i].landms_x2) right = detections[i].landms_x2;
            if (bot   < detections[i].landms_y4) bot   = detections[i].landms_y4;
            int width = right - left;
            int height = bot - top;
            if (width < min_lp_width) continue; //宽度过小
            if (height < min_lp_height) continue;//高度过小

            //保存结果
            Result_item_License_Plate_Detection license;
            license.x1            = vehicle.x1;
            license.y1            = vehicle.y1;
            license.x2            = vehicle.x2;
            license.y2            = vehicle.y2;
            license.landms_x1     = detections[i].landms_x1;
            license.landms_x2     = detections[i].landms_x2;
            license.landms_x3     = detections[i].landms_x3;
            license.landms_x4     = detections[i].landms_x4;
            license.landms_y1     = detections[i].landms_y1;
            license.landms_y2     = detections[i].landms_y2;
            license.landms_y3     = detections[i].landms_y3;
            license.landms_y4     = detections[i].landms_y4;
            license.score         = detections[i].score;
            license.car_idx       = vehicle.temp_idx;
            license.license_type  = detections[i].license_type;
            license.license_color = detections[i].license_color;

#ifdef _OPENMP
#pragma omp ordered
#endif
            licenses.push_back(license);
        }
    }
    return true;
};

/// @brief 初始化
/// @param root_dir         模块的配置目录
bool Alg_Module_License_Plate_Detection::init_from_root_dir(std::string root_dir)
{
    bool load_res = true;

    //加载模块配置文件
    load_res = this->load_module_cfg(root_dir + "/cfgs/" + this->node_name + "/module_cfg.xml");
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load module cfg failed",this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }

    std::shared_ptr<Module_cfg_License_Plate_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Detection>(this->get_module_cfg());

    //如果文件中有运行频率的字段，则使用文件中设定的频率
    int tick_interval;
    load_res = module_cfg->get_int("tick_interval", tick_interval);
    if (load_res) 
        this->tick_interval_ms = tick_interval_ms;
    else 
        this->tick_interval_ms = 100;

    //加载模型相关参数
    load_res = module_cfg->get_string("model_path", this->model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("model_name", this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("model_cfg_path", this->model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    //加载模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name+"/" + this->model_cfg_path, this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model_cfg failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    
    //加载模型
    load_res = this->load_model(root_dir + "/models/" + this->model_path , this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }

    return true;
};

/// @brief 推理
/// @param channel_name         通道名称
/// @param input['image']       输入：原始图片
/// @param input['vehicle']     输入：车辆目标
/// @param output['license']    输出：车牌目标
bool Alg_Module_License_Plate_Detection::forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output) 
{
    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t no vehicle in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }


    std::shared_ptr<Model_cfg_License_Plate_Detection> model_cfg = std::dynamic_pointer_cast<Model_cfg_License_Plate_Detection>(this->get_model_cfg(this->model_name));
    std::shared_ptr<Module_cfg_License_Plate_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Detection>(this->get_module_cfg());


    auto &vehicles = input["vehicle"]->data.detect;
    std::vector<Result_item_License_Plate_Detection> licenses; 
    this->detect_license_plate(channel_name, input, vehicles, licenses);


    //整理检测结果
    auto forward_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_license_t);
    auto &results = forward_output->data.detect_license;
    results.resize(licenses.size());
    for (int i = 0; i < licenses.size(); i++) 
    {   
        results[i].x1            = licenses[i].x1;
        results[i].y1            = licenses[i].y1;
        results[i].x2            = licenses[i].x2;
        results[i].y2            = licenses[i].y2;
        results[i].landms_x1     = licenses[i].landms_x1;
        results[i].landms_x2     = licenses[i].landms_x2;
        results[i].landms_x3     = licenses[i].landms_x3;
        results[i].landms_x4     = licenses[i].landms_x4;
        results[i].landms_y1     = licenses[i].landms_y1;
        results[i].landms_y2     = licenses[i].landms_y2;
        results[i].landms_y3     = licenses[i].landms_y3;
        results[i].landms_y4     = licenses[i].landms_y4;
        results[i].score         = licenses[i].score;
        results[i].temp_idx      = licenses[i].car_idx;
        results[i].car_idx       = licenses[i].car_idx;
        results[i].license_type  = licenses[i].license_type;
        results[i].license_color = licenses[i].license_color;
    }
    output.clear();
    output["license"] = forward_output;
    return true;
};

/// @brief 过滤
/// @param channel_name         通道名称
/// @param input['vehicle']     输入：车辆目标
/// @param input['license']     输入：车牌目标
/// @param output['result]      输出：车辆+车牌关联后的结果
bool Alg_Module_License_Plate_Detection::filter(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>> &input,std::map<std::string,std::shared_ptr<InputOutput>> &output) 
{
    //检查是否包含需要的数据
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t find no vehicle in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }
    if (input.find("license") == input.end()) {
        throw Alg_Module_Exception("Error:\t find no license in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    //未过滤的检测结果
    auto &vehicles = input["vehicle"]->data.detect;
    auto &licenses = input["license"]->data.detect_license;

    //整理检测数据
    auto filter_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_license_t); 
    auto &filter_results = filter_output->data.detect_license;
    for (auto &vehicle : vehicles) {   
        int license_index = 0;
        for (; license_index < licenses.size(); ++license_index)
        {
            if (vehicle.temp_idx == licenses[license_index].car_idx) break;
        }
        
        if (license_index >= licenses.size()) continue;

        Result_Detect_license temp_result;
        temp_result.x1 = vehicle.x1;
        temp_result.y1 = vehicle.y1;
        temp_result.x2 = vehicle.x2;
        temp_result.y2 = vehicle.y2;
        temp_result.tag = vehicle.tag;
        temp_result.temp_idx = vehicle.temp_idx;
        temp_result.car_idx = vehicle.temp_idx;
        temp_result.landms_x1 = licenses[license_index].landms_x1;
        temp_result.landms_x2 = licenses[license_index].landms_x2;
        temp_result.landms_x3 = licenses[license_index].landms_x3;
        temp_result.landms_x4 = licenses[license_index].landms_x4;
        temp_result.landms_y1 = licenses[license_index].landms_y1;
        temp_result.landms_y2 = licenses[license_index].landms_y2;
        temp_result.landms_y3 = licenses[license_index].landms_y3;
        temp_result.landms_y4 = licenses[license_index].landms_y4;
        temp_result.license_type = licenses[license_index].license_type;
        temp_result.license_color = licenses[license_index].license_color;
        temp_result.license = licenses[license_index].license;
        filter_results.push_back(temp_result);
    }
    output.clear();
    output["result"] = filter_output;

    return true;
};

/// @brief 可视化
/// @param channel_name             通道名称
/// @param input['image']           原始图片
/// @param filter_output['result]   车辆+车牌关联后的结果
bool Alg_Module_License_Plate_Detection::display(std::string channel_name,std::map<std::string,std::shared_ptr<InputOutput>> &input,std::map<std::string,std::shared_ptr<InputOutput>> &filter_output) 
{
    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }
    if (filter_output.find("result") == filter_output.end()) {
        throw Alg_Module_Exception("Error:\t no result in filter_output",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    std::shared_ptr<Module_cfg_License_Plate_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Detection>(this->get_module_cfg());

    //加载目标框相关参数
    bool load_res = true;
    int box_color_blue;
    int box_color_green;
    int box_color_red;
    int box_thickness;
    load_res &= module_cfg->get_int("box_color_blue", box_color_blue);
    load_res &= module_cfg->get_int("box_color_green", box_color_green);
    load_res &= module_cfg->get_int("box_color_red", box_color_red);
    load_res &= module_cfg->get_int("box_thickness", box_thickness);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load params failed",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

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

    //车牌事件图片
    auto &results = filter_output["result"]->data.detect_license;
    for (auto &result : results)
    {   
        if (result.license_type == Result_Detect_license::License_Type::Type_UNKNOWN && result.license_color == Result_Detect_license::License_Color::Color_UNKNOWN)
        {
            continue;
        }

        cv::Mat image_copy = image.clone();

        std::vector<cv::Point> pts;
        pts.push_back(cv::Point(result.landms_x1, result.landms_y1));
        pts.push_back(cv::Point(result.landms_x2, result.landms_y2));
        pts.push_back(cv::Point(result.landms_x3, result.landms_y3));
        pts.push_back(cv::Point(result.landms_x4, result.landms_y4));
        cv::polylines(image_copy, pts, true, cv::Scalar(box_color_blue, box_color_green, box_color_red), 1);

        int left  = result.landms_x1;
        int top   = result.landms_y1;
        int right = result.landms_x3;
        int bot   = result.landms_y3;

        if (left  > result.landms_x4) left  = result.landms_x4;
        if (top   > result.landms_y2) top   = result.landms_y2;
        if (right < result.landms_x2) right = result.landms_x2;
        if (bot   < result.landms_y4) bot   = result.landms_y4;

        int x = result.x1 + left;
        int y = result.y1 + top;
        int w = right - left;
        int h = bot - top;

        cv::Rect box(x, y, w, h);
        cv::rectangle(image_copy, box, cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);
        cv::Mat license_plate_image = image(cv::Rect(x, y, w, h));

        result.res_images.insert({"event_image", image_copy});
        result.res_images.insert({"license_plate_image", license_plate_image});
    }
    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_License_Plate_Detection::load_module_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_License_Plate_Detection>(this->node_name);
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Model_cfg_base> Alg_Module_License_Plate_Detection::load_model_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_License_Plate_Detection>();
    res->from_file(cfg_path);
    return res;
};

extern "C" Alg_Module_Base *create()
{
    return new Alg_Module_License_Plate_Detection();                     //返回当前算法模块子类的指针
};
extern "C" void destory(Alg_Module_Base *p)
{
    delete p;
};
