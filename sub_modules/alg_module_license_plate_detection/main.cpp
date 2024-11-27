/*
    项目编译方法
    rm -rf build & cmake -B build & cd build & make & ./alg_module_sample_main
*/

#include "alg_module_detect_tracking.h"
#include "alg_module_license_plate_detection.h"
#include <iostream>
#include <filesystem>

//车辆目标检测和追踪
typedef Alg_Module_Detect_Tracking* (*Create_Detect_Tracking_func)();
typedef void (*Destory_Detect_Tracking_func)(Alg_Module_Detect_Tracking*);
Create_Detect_Tracking_func create_detect_tracking = nullptr;
Destory_Detect_Tracking_func destory_detect_tracking = nullptr;

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

cv::Mat drawLandms(cv::Mat image, int landms_x1, int landms_y1, int landms_x2, int landms_y2, int landms_x3, int landms_y3, int landms_x4, int landms_y4) 
{
    cv::Point2d p1(landms_x1, landms_y1);
    cv::Point2d p2(landms_x2, landms_y2);
    cv::Point2d p3(landms_x3, landms_y3);
    cv::Point2d p4(landms_x4, landms_y4);

    cv::circle(image, p1, 5, cv::Scalar(255, 0, 0), -1, cv::LINE_8);    // 蓝 绿 红
    cv::circle(image, p2, 5, cv::Scalar(0, 255, 0), -1, cv::LINE_8);
    cv::circle(image, p3, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_8);
    cv::circle(image, p4, 5, cv::Scalar(255, 255, 0), -1, cv::LINE_8);

    return image;
}

Alg_Module_Detect_Tracking* get_alg_module_detect_tracking(std::string root_dir, std::string dl_file)
{
    void *handle = nullptr;
    handle = dlopen(dl_file.c_str(), RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "2 %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    *(void **) (&create_detect_tracking) = dlsym(handle, "create");
    *(void **) (&destory_detect_tracking) = dlsym(handle, "destory");

    Alg_Module_Detect_Tracking* module = create_detect_tracking();

    module->init_from_root_dir(root_dir);               //初始化，加载模型和配置文件
    module->get_module_cfg()->get_input_output_cfg();   //可以获取输入输出配置文件
    module->get_module_cfg()->get_publish_cfg();        //可以获取发布的topic的配置文件

    // 请求设备


    std::vector<std::shared_ptr<Device_Handle>> device_handles;                        //获取tpu的核心的handle
    device_handles.resize(1);
    for (long unsigned int i=0; i<device_handles.size(); i++) 
    {
        device_handles[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
    }

    // 增加模型实例
    module->set_device_handles(device_handles);                 //在模块中增加设备句柄

    auto utils = module->check_model_util();
    for (auto iter = utils.begin(); iter != utils.end(); iter++) 
    {
        if(iter->second<0 || iter->second>0.8) 
        {
            for(long unsigned int i=0; i<device_handles.size(); i++)
                module->increase_model_instane(iter->first, i);
        }
    }
    return module;
};

Alg_Module_License_Plate_Detection* get_alg_module_license_plate_detection(std::string root_dir)
{
    // 模型初始化
    Alg_Module_License_Plate_Detection* module = new Alg_Module_License_Plate_Detection();
    module->init_from_root_dir(root_dir);                           //初始化，加载模型和配置文件
    module->get_module_cfg()->get_input_output_cfg();               //可以获取输入输出配置文件
    module->get_module_cfg()->get_publish_cfg();                    //可以获取发布的topic的配置文件

    // 请求设备
    std::vector<std::shared_ptr<Device_Handle>> device_handles;                        //获取tpu的核心的handle
    device_handles.resize(1);
    for (long unsigned int i=0; i<device_handles.size(); i++) 
    {
        device_handles[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
//        bm_dev_request(&device_handles[i], i);
    }

    // 增加模型实例
    module->set_device_handles(device_handles);                     //在模块中增加设备句柄
    auto utils = module->check_model_util();                        //
    for (auto iter = utils.begin(); iter != utils.end(); iter++) 
    {
        if(iter->second<0 || iter->second>0.8) 
        {
            for(long unsigned int i=0; i<device_handles.size(); i++)
                module->increase_model_instane(iter->first, i);
        }
    }

    return module;
}

void test_image_set(std::string image_set_path)
{
    std::string channel_name = "test";

    //车牌检测
    std::string license_plate_detection_root_dir = "./requirement";
    Alg_Module_License_Plate_Detection* module_license_plate_detection = get_alg_module_license_plate_detection(license_plate_detection_root_dir);
    
    //结果保存
    std::string event_save_path = "./requirement/results/event";

    //保存事件图片
    if (access(event_save_path.c_str(), F_OK) == 0) {
        //清空文件夹下的检测结果
        std::string cmd = "rm " + event_save_path + "/*";
        system(cmd.c_str());
    } else {
        //创建保存事件发生时截图的图片
        std::string cmd = "mkdir " + event_save_path;
        system(cmd.c_str());
    } 

    //获取所有图片的路径
    std::vector<std::string> image_names;
    for (const auto& entry : std::filesystem::directory_iterator(image_set_path)) 
    {   
        image_names.push_back(entry.path().filename().string());
    }

    std::shared_ptr<Device_Handle> handle(get_device_handle(0));

    for (int i = 0; i < image_names.size(); i++)
    {
        std::cout << std::to_string(i+1) << "/" << std::to_string(image_names.size());

        cv::Mat image = cv::imread(image_set_path + "/" + image_names[i]);
        std::shared_ptr<QyImage> input_image=from_mat(image,handle);


        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image = input_image;

        auto vehicle_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
        vehicle_output->data.detect.resize(1);
        auto& vehicle_results = vehicle_output->data.detect;
        vehicle_results[0].x1        = 0;
        vehicle_results[0].y1        = 0;
        vehicle_results[0].x2        = image.cols+10;
        vehicle_results[0].y2        = image.rows+10;
        vehicle_results[0].score     = 0;
        vehicle_results[0].class_id  = 0;
        vehicle_results[0].temp_idx  = 0;
       
        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        input["image"] = input_data_image;
        input["vehicle"] = vehicle_output;
        module_license_plate_detection->forward(channel_name, input, output);
        output["vehicle"] = input["vehicle"];
        module_license_plate_detection->filter(channel_name, output, filter_output);
        module_license_plate_detection->display(channel_name, input, filter_output);


        //保存车牌事件检测结果
        auto& results = filter_output["result"]->data.detect_license;
        std::cout << " 检测结果数量:" << std::to_string(results.size());
        if (results.size() > 0)
        {
            std::string event_image_save_path = event_save_path + "/" +  image_names[i];
            cv::imwrite(event_image_save_path, results[0].res_images["event_image"]);
        }

        std::cout << std::endl;
    }  
};

void test_video(std::string video_path)
{ 
    std::string channel_name = "test";

    //车牌检测
    std::string root_dir = "./requirement";
    Alg_Module_License_Plate_Detection* module = get_alg_module_license_plate_detection(root_dir);
    
    //视频文件,结果保存
    std::string video_save_path = "./requirement/results/result.mp4";
    std::string event_save_path = "./requirement/results/event";

    //打开视频
    cv::VideoCapture video;
    video.open(video_path);
    int frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);

    //保存视频
    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat image(height, width, CV_8UC3);

    //保存事件图片
    int event_count = 0;
    if (access(event_save_path.c_str(), F_OK) == 0) {
        //清空文件夹下的检测结果
        std::string cmd = "rm " + event_save_path + "/*";
        system(cmd.c_str());
    } else {
        //创建保存事件发生时截图的图片
        std::string cmd = "mkdir " + event_save_path;
        system(cmd.c_str());
    } 

    std::shared_ptr<Device_Handle> handle(get_device_handle(0));

    for (int i = 0; i < frame_count; ++i)
    {
        video.read(image);
        
        std::cout << std::to_string(i+1) << "/" << std::to_string(frame_count);

        std::shared_ptr<QyImage> input_image=from_mat(image,handle);

        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image = input_image;

        auto vehicle_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
        vehicle_output->data.detect.resize(1);
        auto& vehicle_results = vehicle_output->data.detect;
        vehicle_results[0].x1        = 0;
        vehicle_results[0].y1        = 0;
        vehicle_results[0].x2        = image.cols-1;
        vehicle_results[0].y2        = image.rows-1;
        vehicle_results[0].score     = 0;
        vehicle_results[0].class_id  = 0;
        vehicle_results[0].temp_idx  = 0;
       
        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        input["image"] = input_data_image;
        input["vehicle"] = vehicle_output;
        module->forward(channel_name, input, output);
        output["vehicle"] = input["vehicle"];
        module->filter(channel_name, output, filter_output);
        module->display(channel_name, input, filter_output);


        for (auto &result: filter_output["result"]->data.detect_license) {      
            event_count += 1;

            std::string event_image_save_path = event_save_path + "/" + std::to_string(event_count) + "_event.png";
            std::string plate_image_save_path = event_save_path + "/" + std::to_string(event_count) + "_plate.png";
            
            cv::imwrite(event_image_save_path, result.res_images["event_image"]);
            cv::imwrite(plate_image_save_path, result.res_images["license_plate_image"]);
        }

        std::cout << std::endl;
    }
};

int main(int argc, char* argv[])
{
    test_video("/home/vensin/workspace_sun/test_data/车牌检测_测试视频.mp4");
    return 0;
}
