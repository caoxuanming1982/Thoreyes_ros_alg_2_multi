#include "alg_module_license_plate_recognition.h"
#include <iostream>

void roi_pooling(Output &net_output_feature,std::vector<Result_item_License_Plate_Recognition> &output,int img_h,int img_w)
{
    float* features=(float*)net_output_feature.data.data();
    int f_c=net_output_feature.shape[0];
    int f_h=net_output_feature.shape[1];
    int f_w=net_output_feature.shape[2];
    float factor_h=1.0*f_h/img_h;
    float factor_w=1.0*f_w/img_w;
    int f_size=f_h*f_w;
    for(int i=0;i<output.size();i++){
        output[i].feature.clear();
        int x2=int(output[i].x2*factor_w+1);
        int y2=int(output[i].y2*factor_h+1);
        int x1=int(output[i].x1*factor_w);
        int y1=int(output[i].y1*factor_h);
        float sub=(y2-y1)*(x2-x1);
        output[i].feature.resize(f_c);
        for(int c=0;c<f_c;c++){
            float val=0;
            int offset_c=c*f_size;
            for(int h=y1;h<y2;h++){
                int offset_h=h*f_w;
                for(int w=x1;w<x2;w++){
                    val+=features[w+offset_h+offset_c];
                }
            }
            if(isinf(val))
                val=1.8e17;
            else if(isinf(val)==-1)
                val=-1.8e17;
            else if(isnan(val))
                val=0;
            else{
                val=val/sub;
                if(val>1.8e17)
                    val=1.8e17;
                else if(val<-1.8e17)    
                    val=-1.8e17;

            }
            output[i].feature[c]=val;
        }
    }
};
float iou(Result_item_License_Plate_Recognition &box1, Result_item_License_Plate_Recognition &box2)
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
bool sort_score(Result_item_License_Plate_Recognition &box1, Result_item_License_Plate_Recognition &box2)
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

void decode_single_license(float* res, int dim0, int dim1, string &res_label, float &res_score)
{
    std::vector<string> license_classes = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",  
        "V", "W", "X", "Y", "Z",  
        "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂",  
        "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",  
        "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航"
    };
    int last = license_classes.size();
    float score_sum = 0;
    int res_num = 0;
    int num_start_idx = -1;
    int last_province = -1;
    float last_province_score = 0;
    for (int i = 0; i < dim1; i++) 
    {
        float score = res[i];
        int label = 0;
        for (int j = 1;j < dim0;j++) 
        {
            if (score < res[i + j * dim1]) 
            {
                label = j;
                score = res[i + j * dim1];
            }
        }
        if (label > license_classes.size() || label == last) 
        {
            last = license_classes.size();
            continue;
        }
        else 
        {
            if (label > 33 && label < license_classes.size()) 
            {
                if (last_province < 0) 
                {
                    last_province = label;
                    last_province_score = score;
                }
                else 
                {
                    if (last_province_score < score) 
                    {
                        last_province = label;
                        last_province_score = score;
                    }
                    num_start_idx = i + 1;
                    break;
                }
            }
            if (label <= 33) 
            {
                num_start_idx = i;
                break;
            }
        }
    }

    if (last_province >= 0) {
        res_label.append(license_classes[last_province]);
        score_sum += last_province_score;
        last = last_province;
        res_num++;
    }

    for (int i = num_start_idx; i < dim1; i++) 
    {
        float score = res[i];
        int label = 0;
        for (int j = 1; j < dim0; j++) 
        {
            if (score < res[i + j * dim1]) 
            {
                label = j;
                score = res[i + j * dim1];
            }
        }
        if (label > license_classes.size() || label == last) 
        {
            last = license_classes.size();
            continue;
        }
        else 
        {
            res_label.append(license_classes[label]);
            score_sum += score;
            res_num++;
            last = label;
        }
    }
    res_score = score_sum / res_num;
};
void decode_license(float* res, int dim0, int dim1, string &res_label, float &res_score) 
{
    std::vector<string> chars = {  
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", 
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", 
    "W", "X", "Y", "Z", "京", "沪", "津", "渝", "冀", "晋",
    "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂",  
    "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",  
    "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航"  
    };

    // 实现将二维float数组转为 vector
    std::vector<std::vector<float>> transposed_matrix(dim0, std::vector<float>(dim1));  
    for (int i = 0; i < dim0; ++i) {  
        for (int j = 0; j < dim1; ++j) {  
            transposed_matrix[i][j] = res[i * dim1 + j];  
        }  
    }
    std::vector<std::vector<float>> pred(dim1, std::vector<float>(dim0));
    // 执行转置
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            pred[j][i] = transposed_matrix[i][j];
        }
    }
    // argmax操作
    std::string results = "";
    float confidence = 0.0;
    std::vector<int> table_pred;
    for (const auto &row : pred) {
        int max_index = 0;
        float max_value = row[0];
        for (int i = 1; i < row.size(); i++) {
            if (row[i] > max_value) {
                max_index = i;
                max_value = row[i];
            }
        }
        table_pred.push_back(max_index);
    }
    // 取出车牌字符和置信度
    for (int i = 0; i < table_pred.size(); i++) {
        if (table_pred[i] < chars.size() && (i == 0 || (table_pred[i] != table_pred[i - 1]))) {
//            std::cout<<table_pred[i]<<std::endl;
            results.append(chars[table_pred[i]]);
            confidence += pred[i][table_pred[i]];
  //           std::cout << pred[i][table_pred[i]] << std::endl;
        }
    }
    res_label = results;
    res_score = confidence / results.length();
};
bool isValidLicensePlate(std::string &plateNumber,Result_Detect_license license)
{
//    std::cout<<plateNumber.length()<<" "<<plateNumber<<std::endl;
    // 检查位数

    if(license.license_color==Result_Detect_license::Green){
        if (plateNumber.length() != 10 ) {
            return false;
        }
    }
    else if(license.license_color==Result_Detect_license::Blue){
        if (plateNumber.length() != 9 ) {
            return false;
        }
    }    
    else if(license.license_color==Result_Detect_license::Yellow){
        if (plateNumber.length() != 9 &&plateNumber.length() != 11 ) {
            return false;
        }
    }   
    else{
        if (plateNumber.length() < 9 || plateNumber.length() > 11) {
            return false;
        }

    } 
    // 检查带挂字
    std::string searchString = "挂";
    if (plateNumber.length() == 11 ) {
        if(plateNumber.find(searchString) == std::string::npos){
            return false;
        }
    }
    else{
        if(plateNumber.find(searchString) != std::string::npos){
            return false;
        }
    }
    
    // 检查第二位英文
    if (plateNumber[3] < 'A' || plateNumber[3] > 'Z'){
        return false;
    }
    
    //检查数字个数是否大于2
    int digitCount = 0;
    for (char c : plateNumber) {
        if (isdigit(c)) digitCount++;
    }
    if (digitCount <= 2)
        return false;
    
    return true;
};

Alg_Module_License_Plate_Recognition::Alg_Module_License_Plate_Recognition():Alg_Module_Base_private("license_plate_recognition")
{    //参数是模块名，使用默认模块名初始化

};
Alg_Module_License_Plate_Recognition::~Alg_Module_License_Plate_Recognition()
{

};

/// @brief 识别单行车牌
std::string Alg_Module_License_Plate_Recognition::recognize_single_license(std::string channel_name, std::shared_ptr<QyImage>& license_image, Result_Detect vehicle, Result_Detect_license license)
{
    std::shared_ptr<Module_cfg_License_Plate_Recognition> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Recognition>(this->get_module_cfg()); 

    float single_thresh_score;
    module_cfg->get_float("single_thresh_score", single_thresh_score);

    auto net = this->get_model_instance(this->single_model_name);  //获取指定的模型实例 
    if (net == nullptr)
    {   //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        throw Alg_Module_Exception("Error:\t single_model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);       
    }

    auto input_shapes = net->get_input_shapes();                        
    if (input_shapes.size() <= 0)   //判断模型是否已经加载
    {   //获取模型推理实例异常，一般是因为模型实例还未创建
        throw Alg_Module_Exception("Warning:\t single_model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);
    }
    std::vector<Output> net_output;                                         //这一部分与原版本不一样
    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(license_image);
    net->forward(net_input, net_output);

    float* res = (float*)net_output[0].data.data();
    int dim1 = net_output[0].shape[0];
    int dim2 = net_output[0].shape[1];
    std::string label = "";
    float score = 0;
    decode_single_license(res, dim1, dim2, label, score);

    if (score < single_thresh_score) return "";
    
    if (isValidLicensePlate(label,license)) {
        return label;
    } else {
        return "";
    }
};

/// @brief 识别双行车牌
std::string Alg_Module_License_Plate_Recognition::recognize_double_license(std::string channel_name, std::shared_ptr<QyImage>& license_image, Result_Detect vehicle, Result_Detect_license license)
{
    std::shared_ptr<Module_cfg_License_Plate_Recognition> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Recognition>(this->get_module_cfg()); 

    float double_thresh_score;
    module_cfg->get_float("double_thresh_score", double_thresh_score);

    auto net = this->get_model_instance(this->model_name);  //获取指定的模型实例 
    if (net == nullptr)
    {   //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);       
    }

    auto input_shapes = net->get_input_shapes();                        
    if (input_shapes.size() <= 0)   //判断模型是否已经加载
    {   //获取模型推理实例异常，一般是因为模型实例还未创建
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);
    }

    float license_w =license_image->get_width();
    float license_h =license_image->get_height();
    std::shared_ptr<QyImage> sub_image=license_image;


    //1.需要补高
    if (license_w / license_h > 1.75)
    {   //计算上下增加量
        int delta_h = license_w / 1.75 - license_h;
        int add_h_up,add_h_down;
        if ((int)delta_h % 2 != 0) {
            add_h_up = (int)(delta_h / 2);
            add_h_down = (int)(delta_h / 2) + 1;
        } else {
            add_h_up = (int)(delta_h / 2);
            add_h_down = (int)(delta_h / 2);
        }
        sub_image=sub_image->padding(0,0,add_h_up,add_h_down,0);
        sub_image=sub_image->resize(140,80);
        sub_image=sub_image->cvtcolor(true);
    }
    //2.需要补宽
    else
    {
        //计算左右增加量
        int delta_w = license_h * 1.75 - license_w;
        int add_w_left,add_w_right;
        if ((int)delta_w % 2 != 0)
        {
            add_w_left = (int)(delta_w / 2);
            add_w_right = (int)(delta_w / 2) + 1;
        }
        else
        {
            add_w_left = (int)(delta_w / 2);
            add_w_right = (int)(delta_w / 2);
        }
                    
        //定义大padding mat
        int bm_image_tmp_hight = license_h;
        int bm_image_tmp_width = license_w + add_w_left + add_w_right;

        sub_image=sub_image->padding(add_w_left,add_w_right,0,0,0);
        sub_image=sub_image->resize(140,80);
        sub_image=sub_image->cvtcolor(true);


    }

    std::vector<Output> net_output;                                         //这一部分与原版本不一样
    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(sub_image);
    net->forward(net_input, net_output);                                    //这一部分与原版本不一样


    float* _net_res = (float*)net_output[0].data.data();
    int dim1 = net_output[0].shape[1];
    int dim2 = net_output[0].shape[2];
    std::string label = "";
    float score = 0;
    decode_license(_net_res, dim1, dim2, label, score);


    if (score < double_thresh_score) return "";

    if (isValidLicensePlate(label,license)) {
        return label;
        // licenses[license_id].license = label;
        // std::cout << label << std::endl;
    } else {
        // licenses[license_id].license = "";
        return "";
    }
};


std::shared_ptr<QyImage> Alg_Module_License_Plate_Recognition::get_sub_image_with_perspect(std::string channel_name, std::shared_ptr<QyImage>& input,Result_Detect& vehicle, Result_Detect_license& license,int w,int h){

    int width=input->get_width();
    int height=input->get_height();
    

    std::vector<cv::Point2f> coordinate(4);
    coordinate[0].x=license.landms_x1 + vehicle.x1;
    coordinate[1].x = license.landms_x2 + vehicle.x1;
    coordinate[2].x = license.landms_x4 + vehicle.x1;
    coordinate[3].x = license.landms_x3 + vehicle.x1;
    coordinate[0].y = license.landms_y1 + vehicle.y1;
    coordinate[1].y = license.landms_y2 + vehicle.y1;
    coordinate[2].y = license.landms_y4 + vehicle.y1;
    coordinate[3].y = license.landms_y3 + vehicle.y1;
    int license_www = coordinate[1].x - coordinate[0].x;
    int license_hhh = coordinate[2].y - coordinate[0].y;

    if((license_www < 8) || (license_www > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.width invalid " << width << std::endl;
        return std::shared_ptr<QyImage>();
    }
    if((license_hhh < 8) || (license_hhh > 8192)){
        std::cout << "ch " << channel_name << " model " <<  model_name << " input_image.height invalid " << height << std::endl;
        return std::shared_ptr<QyImage>();
    }
    if(w>0)
        license_www=w;
    if(h>0)
        license_hhh=h;

    std::shared_ptr<QyImage> res=input->warp_perspective(coordinate,license_www,license_hhh);


    return res;

};
std::shared_ptr<QyImage> Alg_Module_License_Plate_Recognition::get_image(std::map<std::string, std::shared_ptr<InputOutput>>& input){

    std::shared_ptr<QyImage> image;

    if(input["image"]->data_type==InputOutput::Type::Image_t){
        image=input["image"]->data.image;
        if(image==nullptr){
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return std::shared_ptr<QyImage>();
    }
    return image;

};


/// @brief 识别车牌
bool Alg_Module_License_Plate_Recognition::recognize_license_plate(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, 
    std::vector<Result_Detect> &vehicles, std::vector<Result_Detect_license> &licenses, std::vector<Result_item_License_Plate_Recognition> &licenses_res) 
{
    std::shared_ptr<Module_cfg_License_Plate_Recognition> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Recognition>(this->get_module_cfg());         //获取模块配置

    bool load_res = true;
    float single_thresh_score;
    float double_thresh_score;
    load_res &= module_cfg->get_float("single_thresh_score", single_thresh_score);
    load_res &= module_cfg->get_float("double_thresh_score", double_thresh_score);
    if (load_res == false)
    {   //找不到必要的配置参数，检查配置文件是否有对应的字段，检查类型，检测名称
        throw Alg_Module_Exception("Error:\t load module params failed",this->node_name,Alg_Module_Exception::Stage::inference);         
        return false;
    }

    std::shared_ptr<QyImage> input_image=get_image(input);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int license_id = 0; license_id < licenses.size(); license_id++)
    {   
        //寻找车辆的索引编号
        int vehicle_id;
        for (vehicle_id = 0; vehicle_id < vehicles.size(); vehicle_id++)
            if (vehicles[vehicle_id].temp_idx == licenses[license_id].car_idx) break;

        if (vehicle_id == vehicles.size()) continue; //有车牌但没车辆

        std::shared_ptr<QyImage> sub_image;
        
        std::string label;
            // 双行车牌d
        if (licenses[license_id].license_type == Result_Detect_license::Double &&licenses[license_id].license_color==Result_Detect_license::Yellow)
        {
            sub_image= get_sub_image_with_perspect(channel_name,input_image,vehicles[vehicle_id], licenses[license_id]);
            if(sub_image==nullptr)
            {
                label="";
            }    
            else{

                label = this->recognize_double_license(channel_name, sub_image, vehicles[vehicle_id], licenses[license_id]);
            }
            
        }
        else
        {
            sub_image= get_sub_image_with_perspect(channel_name,input_image,vehicles[vehicle_id], licenses[license_id],160,40);
           if(sub_image==nullptr)
            {
                label="";
            }        
            else{
                label = this->recognize_single_license(channel_name, sub_image, vehicles[vehicle_id], licenses[license_id]);
            }
        }

        // 单行车牌

        // // TODO 未知类型车牌
        // if (licenses[license_id].license_type == Result_Detect_license::Type_UNKNOWN) continue;

        // if (label.size() == 0) continue;

        Result_item_License_Plate_Recognition license;
        license.x1              = licenses[license_id].x1;
        license.y1              = licenses[license_id].y1;
        license.x2              = licenses[license_id].x2;
        license.y2              = licenses[license_id].y2;
        license.landms_x1       = licenses[license_id].landms_x1;
        license.landms_x2       = licenses[license_id].landms_x2;
        license.landms_x3       = licenses[license_id].landms_x3;
        license.landms_x4       = licenses[license_id].landms_x4;
        license.landms_y1       = licenses[license_id].landms_y1;
        license.landms_y2       = licenses[license_id].landms_y2;
        license.landms_y3       = licenses[license_id].landms_y3;
        license.landms_y4       = licenses[license_id].landms_y4;
        license.score           = licenses[license_id].score;
        license.car_idx         = licenses[license_id].car_idx;
        license.license_type    = licenses[license_id].license_type;
        license.license_color   = licenses[license_id].license_color;
        license.license         = label;
//        std::cout<<license.str()<<std::endl;

#ifdef _OPENMP
#pragma omp ordered
#endif
        licenses_res.push_back(license);
    }

    return true;
};

/// @brief 初始化
bool Alg_Module_License_Plate_Recognition::init_from_root_dir(std::string root_dir)
{
    bool load_res = true;

    //加载模块配置文件
    load_res = this->load_module_cfg(root_dir + "/cfgs/" + this->node_name + "/module_cfg.xml");
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load module cfg failed",this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }

    std::shared_ptr<Module_cfg_base> module_cfg = this->get_module_cfg();   //获取模块配置

    //如果文件中有运行频率的字段，则使用文件中设定的频率
    int tick_interval;
    load_res = module_cfg->get_int("tick_interval", tick_interval);
    if (load_res) 
        this->tick_interval_ms = tick_interval_ms;
    else 
        this->tick_interval_ms = 100;

    //加载模型相关参数
    load_res = module_cfg->get_string("single_model_name", this->single_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no single_model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("single_model_path", this->single_model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no single_model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("single_model_cfg_path", this->single_model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no single_model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
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

    // 加载模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name +"/"+ this->single_model_cfg_path, this->single_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load single_model_cfg_path failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name +"/"+ this->model_cfg_path, this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model_cfg failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    
    //加载模型
    load_res = this->load_model(root_dir + "/models/" + this->single_model_path , this->single_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load single_model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }
    load_res = this->load_model(root_dir + "/models/" + this->model_path , this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }

    return true;
};

/// @brief 推理
bool Alg_Module_License_Plate_Recognition::forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output) 
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
    if (input.find("license") == input.end()) {
        throw Alg_Module_Exception("Error:\t no license in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }

    std::shared_ptr<Module_cfg_License_Plate_Recognition> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Recognition>(this->get_module_cfg());         //获取模块配置
    
    auto &vehicles = input["vehicle"]->data.detect;                     //车辆
    auto &licenses = input["license"]->data.detect_license;             //车牌
    std::vector<Result_item_License_Plate_Recognition> licenses_res;    //检测出的车牌结果

    this->recognize_license_plate(channel_name, input, vehicles, licenses, licenses_res);

    //推理完成后，自己创建的bm_image需要自己销毁

    //整理检测结果
    auto forward_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_license_t); //结果数据的结构
    auto &forward_results = forward_output->data.detect_license;
    forward_results.resize(licenses_res.size());
    for (int i = 0; i < licenses_res.size(); i++) //循环填充结果数据
    {   
        forward_results[i].x1            = licenses_res[i].x1;
        forward_results[i].y1            = licenses_res[i].y1;
        forward_results[i].x2            = licenses_res[i].x2;
        forward_results[i].y2            = licenses_res[i].y2;
        forward_results[i].landms_x1     = licenses_res[i].landms_x1;
        forward_results[i].landms_x2     = licenses_res[i].landms_x2;
        forward_results[i].landms_x3     = licenses_res[i].landms_x3;
        forward_results[i].landms_x4     = licenses_res[i].landms_x4;
        forward_results[i].landms_y1     = licenses_res[i].landms_y1;
        forward_results[i].landms_y2     = licenses_res[i].landms_y2;
        forward_results[i].landms_y3     = licenses_res[i].landms_y3;
        forward_results[i].landms_y4     = licenses_res[i].landms_y4;
        forward_results[i].score         = licenses_res[i].score;
        forward_results[i].temp_idx      = licenses_res[i].car_idx;
        forward_results[i].car_idx       = licenses_res[i].car_idx;
        forward_results[i].license_type  = licenses_res[i].license_type;
        forward_results[i].license_color = licenses_res[i].license_color;
        forward_results[i].license       = licenses_res[i].license;
    }
    output.clear();
    output["license"] = forward_output;
    return true;
};

/// @brief 过滤
bool Alg_Module_License_Plate_Recognition::filter(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output) 
{
    //检查是否包含需要的数据
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t no vehicle in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }
    if (input.find("license") == input.end()) {
        throw Alg_Module_Exception("Error:\t no license in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    //未过滤的检测结果
    auto &forward_results_vehicle = input["vehicle"]->data.detect;
    auto &forward_results_license = input["license"]->data.detect_license;

    //整理检测数据
    auto filter_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_license_t); 
    auto &filter_results = filter_output->data.detect_license;
    for (int i = 0; i < forward_results_vehicle.size(); i++)
    {   
        bool vehicle_has_license = false;
        for (int j = 0; j < forward_results_license.size(); j++)
        {
            if (forward_results_vehicle[i].temp_idx == forward_results_license[j].car_idx && forward_results_license[j].license.length() >= 6)
            {
                vehicle_has_license = true;
            }
        }

        if (vehicle_has_license != true) continue;

        Result_Detect_license temp_result;
        temp_result.x1      = forward_results_vehicle[i].x1;
        temp_result.y1      = forward_results_vehicle[i].y1;
        temp_result.x2      = forward_results_vehicle[i].x2;
        temp_result.y2      = forward_results_vehicle[i].y2;
        temp_result.tag     = forward_results_vehicle[i].tag;
        temp_result.car_idx = forward_results_vehicle[i].temp_idx;

        for (int j = 0; j < forward_results_license.size(); j++)
        {
            if (forward_results_vehicle[i].temp_idx == forward_results_license[j].car_idx)
            {
                temp_result.landms_x1     = forward_results_license[j].landms_x1;
                temp_result.landms_x2     = forward_results_license[j].landms_x2;
                temp_result.landms_x3     = forward_results_license[j].landms_x3;
                temp_result.landms_x4     = forward_results_license[j].landms_x4;
                temp_result.landms_y1     = forward_results_license[j].landms_y1;
                temp_result.landms_y2     = forward_results_license[j].landms_y2;
                temp_result.landms_y3     = forward_results_license[j].landms_y3;
                temp_result.landms_y4     = forward_results_license[j].landms_y4;
                temp_result.license_type  = forward_results_license[j].license_type;
                temp_result.license_color = forward_results_license[j].license_color;
                temp_result.license       = forward_results_license[j].license;
                break;
            }
        }
        filter_results.push_back(temp_result);
    }
    output.clear();
    output["result"] = filter_output;

    return true;
};

/// @brief 可视化
bool Alg_Module_License_Plate_Recognition::display(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &filter_output) 
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

    std::shared_ptr<Module_cfg_License_Plate_Recognition> module_cfg = std::dynamic_pointer_cast<Module_cfg_License_Plate_Recognition>(this->get_module_cfg());

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
        throw Alg_Module_Exception("Error:\t load param failed",this->node_name,Alg_Module_Exception::Stage::display);
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
    for (int i = 0; i < results.size(); i++)
    {   
        cv::Mat image_copy = image.clone();

        int left  = results[i].landms_x1;
        int top   = results[i].landms_y1;
        int right = results[i].landms_x3;
        int bot   = results[i].landms_y3;

        if (left  > results[i].landms_x4) left  = results[i].landms_x4;
        if (top   > results[i].landms_y2) top   = results[i].landms_y2;
        if (right < results[i].landms_x2) right = results[i].landms_x2;
        if (bot   < results[i].landms_y4) bot   = results[i].landms_y4;

        int x = results[i].x1 + left;
        int y = results[i].y1 + top;
        int w = right - left;
        int h = bot - top;

        cv::Rect box(x, y, w, h);
        cv::rectangle(image_copy, box, cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);

        
        cv::Point point(x, y);
        cv::putText(image_copy, results[i].license, point, 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);

        if(results[i].license_color==Result_Detect_license::Blue)
            cv::putText(image_copy, "Blue", point+cv::Point(0,20), 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
        else if (results[i].license_color==Result_Detect_license::Yellow)
            cv::putText(image_copy, "Yellow", point+cv::Point(0,20), 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
        else if (results[i].license_color==Result_Detect_license::Green)
            cv::putText(image_copy, "Green", point+cv::Point(0,20), 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);

        if(results[i].license_type==Result_Detect_license::Single)
            cv::putText(image_copy, "Single", point+cv::Point(0,40), 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
        else if (results[i].license_type==Result_Detect_license::Double)
            cv::putText(image_copy, "Double", point+cv::Point(0,40), 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
        
        cv::circle(image_copy,cv::Point2i(results[i].x1+results[i].landms_x1,results[i].y1+results[i].landms_y1),3,cv::Scalar(255,255,255),2);
        cv::circle(image_copy,cv::Point2i(results[i].x1+results[i].landms_x2,results[i].y1+results[i].landms_y2),3,cv::Scalar(0,255,255),2);
        cv::circle(image_copy,cv::Point2i(results[i].x1+results[i].landms_x3,results[i].y1+results[i].landms_y3),3,cv::Scalar(255,0,255),2);
        cv::circle(image_copy,cv::Point2i(results[i].x1+results[i].landms_x4,results[i].y1+results[i].landms_y4),3,cv::Scalar(255,255,0),2);

        results[i].res_images.insert({"event_image", image_copy});
    }
    
    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_License_Plate_Recognition::load_module_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_License_Plate_Recognition>(this->node_name);
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Model_cfg_base> Alg_Module_License_Plate_Recognition::load_model_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_License_Plate_Recognition>();
    res->from_file(cfg_path);
    return res;
};

extern "C" Alg_Module_Base *create()
{
    return new Alg_Module_License_Plate_Recognition();                     //返回当前算法模块子类的指针
};
extern "C" void destory(Alg_Module_Base *p)
{
    delete p;
};
