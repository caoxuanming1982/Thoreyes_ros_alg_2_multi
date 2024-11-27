/**
 * 过线事件检测
 * CA模型检测J（车辆类型，车辆朝向，车辆颜色）（时序校验）
 * 车辆运动方向检测
 * 车辆所处车道检测
 * */

#include "alg_module_detect_tracking.h"
#include "alg_module_traffic_flow_detection.h"
#include "alg_module_license_plate_detection.h"
#include "alg_module_license_plate_recognition.h"
#include <iostream>
#include <filesystem>
#include <time.h>
#include <dlfcn.h>  

//车辆目标检测和追踪
typedef Alg_Module_Detect_Tracking* (*Create_Detect_Tracking_func)();
typedef void (*Destory_Detect_Tracking_func)(Alg_Module_Detect_Tracking*);
Create_Detect_Tracking_func create_detect_tracking = nullptr;
Destory_Detect_Tracking_func destory_detect_tracking = nullptr;

//车牌检测
typedef Alg_Module_License_Plate_Detection* (*Create_License_Plate_Detection_func)();
typedef void (*Destory_License_Plate_Detection_func)(Alg_Module_License_Plate_Detection*);
Create_License_Plate_Detection_func create_license_plate_detection = nullptr;
Destory_License_Plate_Detection_func destory_license_plate_detection = nullptr;

//车牌识别
typedef Alg_Module_License_Plate_Recognition* (*Create_License_Plate_Recognition_func)();
typedef void (*Destory_License_Plate_Recognition_func)(Alg_Module_License_Plate_Recognition*);
Create_License_Plate_Recognition_func create_license_plate_recognition = nullptr;
Destory_License_Plate_Recognition_func destory_license_plate_recognition = nullptr;

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

Alg_Module_Traffic_Flow_Detection *get_Alg_Module_Traffic_Flow_Detection(std::string root_dir)
{
    // 模型初始化
    Alg_Module_Traffic_Flow_Detection* module_traffic_flow_detection = new Alg_Module_Traffic_Flow_Detection();
    module_traffic_flow_detection->init_from_root_dir(root_dir);                            //初始化，加载模型和配置文件
    module_traffic_flow_detection->get_module_cfg()->get_input_output_cfg();                //可以获取输入输出配置文件
    module_traffic_flow_detection->get_module_cfg()->get_publish_cfg();                     //可以获取发布的topic的配置文件

    // 请求设备
    std::vector<torch::Device> device_handles;                        //获取tpu的核心的handle
    int n_device=torch::cuda::device_count();

    for (long unsigned int i=0; i<n_device; i++) {
        device_handles.push_back(torch::Device(torch::DeviceType::CUDA,i));

    }

    // 增加模型实例
    module_traffic_flow_detection->set_device_handles(device_handles);                      //在模块中增加设备句柄
    auto utils = module_traffic_flow_detection->check_model_util();                         //
    for (auto iter = utils.begin(); iter != utils.end(); iter++) 
    {
        if(iter->second<0 || iter->second>0.8) 
        {
            for(long unsigned int i=0; i<device_handles.size(); i++)
                module_traffic_flow_detection->increase_model_instane(iter->first, i);
        }
    }

    return module_traffic_flow_detection;
};

Alg_Module_Detect_Tracking *get_alg_module_detect_tracking(std::string root_dir, std::string dl_file)
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
    std::vector<torch::Device> device_handles;                        //获取tpu的核心的handle
    int n_device=torch::cuda::device_count();

    for (long unsigned int i=0; i<n_device; i++) {
        device_handles.push_back(torch::Device(torch::DeviceType::CUDA,i));

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

Alg_Module_License_Plate_Detection *get_alg_module_license_plate_detection(std::string root_dir, std::string dl_file)
{
    void *handle = nullptr;
    handle = dlopen(dl_file.c_str(), RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "2 %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    *(void **) (&create_license_plate_detection) = dlsym(handle, "create");
    *(void **) (&destory_license_plate_detection) = dlsym(handle, "destory");

    Alg_Module_License_Plate_Detection* module = create_license_plate_detection();

    module->init_from_root_dir(root_dir);               //初始化，加载模型和配置文件
    module->get_module_cfg()->get_input_output_cfg();   //可以获取输入输出配置文件
    module->get_module_cfg()->get_publish_cfg();        //可以获取发布的topic的配置文件

    // 请求设备
    std::vector<torch::Device> device_handles;                        //获取tpu的核心的handle
    int n_device=torch::cuda::device_count();

    for (long unsigned int i=0; i<n_device; i++) {
        device_handles.push_back(torch::Device(torch::DeviceType::CUDA,i));

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
}

Alg_Module_License_Plate_Recognition *get_alg_module_license_plate_recognition(std::string root_dir, std::string dl_file)
{
    void *handle = nullptr;
    handle = dlopen(dl_file.c_str(), RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "2 %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    *(void **) (&create_license_plate_recognition) = dlsym(handle, "create");
    *(void **) (&destory_license_plate_recognition) = dlsym(handle, "destory");

    Alg_Module_License_Plate_Recognition *module = create_license_plate_recognition();

    module->init_from_root_dir(root_dir);               //初始化，加载模型和配置文件
    module->get_module_cfg()->get_input_output_cfg();   //可以获取输入输出配置文件
    module->get_module_cfg()->get_publish_cfg();        //可以获取发布的topic的配置文件

    // 请求设备
    std::vector<torch::Device> device_handles;                        //获取tpu的核心的handle
    int n_device=torch::cuda::device_count();

    for (long unsigned int i=0; i<n_device; i++) {
        device_handles.push_back(torch::Device(torch::DeviceType::CUDA,i));

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
}

void test_video(std::string video_path, std::string channel_cfg_path, int max_frame = 7500,bool use_gpu=false)
{ 
    //目标检测
    std::string root_dir = "../requirement";
    Alg_Module_Traffic_Flow_Detection* module = get_Alg_Module_Traffic_Flow_Detection(root_dir);
    
    //车牌检测
    std::string license_plate_detection_root_dir = "../../alg_module_license_plate_detection/requirement";
    std::string license_plate_detection_dl_file = "../../alg_module_license_plate_detection/build/libalg_module_license_plate_detection_share.so";
    Alg_Module_License_Plate_Detection* module_license_plate_detection = get_alg_module_license_plate_detection(license_plate_detection_root_dir, license_plate_detection_dl_file);

    //车牌识别
    std::string license_plate_recognition_root_dir = "../../alg_module_license_plate_recognition/requirement";
    std::string license_plate_recognition_dl_file = "../../alg_module_license_plate_recognition/build/libalg_module_license_plate_recognition_share.so";
    Alg_Module_License_Plate_Recognition* module_license_plate_recognition = get_alg_module_license_plate_recognition(license_plate_recognition_root_dir, license_plate_recognition_dl_file);

    //检测与追踪
    std::string detect_tracking_root_dir = "../../alg_module_detect_tracking/requirement";
    std::string detect_tracking_dl_file  = "../../alg_module_detect_tracking/build/libalg_module_detect_tracking_share.so";
    Alg_Module_Detect_Tracking* module_detect_tracking = get_alg_module_detect_tracking(detect_tracking_root_dir, detect_tracking_dl_file);

    //设备和图片

    //保存视频
    int write_width = 1920;
    int write_height = 1080;
    cv::Mat image_write(write_height, write_width, CV_8UC3);
    cv::VideoWriter writer;
    int coder = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 25.0;

    //保存事件图片
    std::string event_save_path = "../requirement/results/event";
    if (access(event_save_path.c_str(), F_OK) == 0) {
        //清空文件夹下的检测结果
        std::string cmd = "rm " + event_save_path + "/*";
        system(cmd.c_str());
    } else {
        //创建保存事件发生时截图的图片
        std::string cmd = "mkdir " + event_save_path;
        system(cmd.c_str());
    } 

    //打开视频
    cv::VideoCapture video;
    video.open(video_path);
    int frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
    frame_count = frame_count < max_frame ? frame_count : max_frame;
    
    //打开图片
    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat image(height, width, CV_8UC3);

    //保存视频
    std::string video_save_path = "../requirement/results/result.mp4";
    writer.open(video_save_path, coder, fps, cv::Size(write_width, write_height), CV_8UC3);

    //初始化模块
    std::string channel_name = "test";
    module->load_channel_cfg(channel_name, channel_cfg_path);
    module->init_channal_data(channel_name);   
    module_detect_tracking->init_channal_data(channel_name);

    float max_cost = 0.;
    float cost = 0.;
    int cost_num = 0;

    std::vector<int> event_count;
    int select_device=0;

    for (int i = 0; i < frame_count; ++i)
    {
        if (i%5 != 0) {
            video.grab();
            continue;
        } else {
            video.read(image);
        };

        std::cout << std::to_string(i+1) << "/" << std::to_string(frame_count);
        std::map<std::string,std::shared_ptr<InputOutput>> input;

        if(use_gpu){
            cv::cuda::setDevice(select_device);
            auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_cv_gpu_t);
            input_data_image->data.image_cv_gpu.device_idx=select_device;
            input_data_image->data.image_cv_gpu.image.upload(image);
            input["image"] = input_data_image;

        }
        else{
            auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_cv_t);
            input_data_image->data.image_cv.image = image.clone();
            input["image"] = input_data_image;


        }

        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "program start" << std::endl;

        module_detect_tracking->forward(channel_name, input, output);           //目标检测, 目标追踪 

        input["vehicle"] = output["vehicle"];
        module_license_plate_detection->forward(channel_name, input, output);   
        input["license"] = output["license"];
        module_license_plate_recognition->forward(channel_name, input, output);

        input["license"] = output["license"];
        module->forward(channel_name, input, output);        //检测
        module->filter(channel_name, output, filter_output); //过滤
        module->display(channel_name, input, filter_output); //可视化

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        cost += elapsed.count();
        if  (elapsed.count() > max_cost) {
            max_cost = elapsed.count();
        }
        cost_num += 1;


        //保存跟踪检测结果
        auto &vehicles = output["vehicle"]->data.detect;
        for (auto &result : vehicles) {   
            image = drawBox(image, result.x1, result.y1, result.x2, result.y2);
        }

        //保存事件检测结果
        auto &results = filter_output["result"]->data.detect_license;
        std::cout << " 车辆目标数量 " << vehicles.size() << " 事件数量 " << results.size();
        for (auto &result : results) {
            std::string text = std::to_string(result.temp_idx) + " ";
            if (result.tag == "车流") text += "count ";
            text += result.license;
            cv::rectangle(image, cv::Rect(result.x1, result.y1, result.x2-result.x1, result.y2-result.y1), cv::Scalar(255, 0, 0), 2);
            cv::putText(image, text, cv::Point(result.x1, result.y1), 1, 2, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
        }
        cv::resize(image, image_write, cv::Size(write_width, write_height));
        writer.write(image_write);

        // 保存车辆事件检测结果
        for (auto &result : results) {   
            while (result.region_idx >= event_count.size()) {
                event_count.push_back(0);
            }
            event_count[result.region_idx]++;

            std::string event_image_save_path = event_save_path + "/" + std::to_string(i) + "_"  + std::to_string(event_count[result.region_idx]) + "_" + result.tag + "_" + result.license;
            event_image_save_path += "_" + result.ext_result["vehicle_type"].tag;
            event_image_save_path += "_" + result.ext_result["vehicle_color"].tag;
            event_image_save_path += "_" + result.ext_result["travel_dir"].tag;
            event_image_save_path += "_" + result.ext_result["vehicle_head_dir"].tag;
            event_image_save_path += "_" + result.ext_result["lane_no"].tag;
            event_image_save_path += ".png";
            cv::imwrite(event_image_save_path, result.res_images["event_image"]);
        }
        for (auto &result : results) {   
            while (result.ext_result["lane_no"].class_id >= event_count.size()) {
                event_count.push_back(0);
            }
            event_count[result.ext_result["lane_no"].class_id]++;
        }
        std::cout << std::endl;
    }
    writer.release();
    
    for (int i = 0; i < event_count.size(); ++i) {
        std::cout << "region" << i << ": " << event_count[i] << std::endl;
    }
    std::cout << "cost: " << cost/cost_num << std::endl;
    std::cout << "max_cost: " << max_cost << std::endl;

}

int main(int argc, char *argv[])
{
    // test_video("/home/vensin/workspace_sun/test_data/车流统计/西藏东线03_D0140_20240402153457至20240402154640.mp4", "/home/vensin/workspace_sun/test_data/all.xml", 1000);

#ifdef HAVE_CUDA
    std::cout<<"use cuda image process"<<std::endl;
    test_video(
        "../../../test_data/车流统计/00010002445000000.mp4", 
        "../../../test_data/车流统计/00010002445000000.xml", 
        1000,true);

#else
    test_video(
        "../../../test_data/车流统计/00010002445000000.mp4", 
        "../../../test_data/车流统计/00010002445000000.xml", 
        1000,false);
#endif        
}
