/*
    项目编译方法
    rm -rf build & cmake -B build & cd build & make & ./alg_module_sample_main
*/

#include "alg_module_bus_detection.h"
#include "alg_module_license_plate_detection.h"
#include "alg_module_license_plate_recognition.h"
#include <iostream>

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

Alg_Module_Bus_Detection* get_alg_module_bus_detection(std::string root_dir)
{
    // 模型初始化
    Alg_Module_Bus_Detection* module_bus_detection = new Alg_Module_Bus_Detection();
    module_bus_detection->init_from_root_dir(root_dir);                           //初始化，加载模型和配置文件
    module_bus_detection->get_module_cfg()->get_input_output_cfg();               //可以获取输入输出配置文件
    module_bus_detection->get_module_cfg()->get_publish_cfg();                    //可以获取发布的topic的配置文件

    // 请求设备
    std::vector<std::shared_ptr<Device_Handle>> device_handles;                        //获取tpu的核心的handle
    device_handles.resize(1);
    for (long unsigned int i=0; i<device_handles.size(); i++) 
    {
        device_handles[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
    }

    // 增加模型实例
    module_bus_detection->set_device_handles(device_handles);                     //在模块中增加设备句柄
    auto utils = module_bus_detection->check_model_util();                        //
    for (auto iter = utils.begin(); iter != utils.end(); iter++) 
    {
        if(iter->second<0 || iter->second>0.8) 
        {
            for(long unsigned int i=0; i<device_handles.size(); i++)
                module_bus_detection->increase_model_instane(iter->first, i);
        }
    }

    return module_bus_detection;
}

Alg_Module_License_Plate_Detection* get_alg_module_license_plate_detection(std::string root_dir, std::string dl_file)
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
}

Alg_Module_License_Plate_Recognition* get_alg_module_license_plate_recognition(std::string root_dir, std::string dl_file)
{
    void *handle = nullptr;
    handle = dlopen(dl_file.c_str(), RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "2 %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    *(void **) (&create_license_plate_recognition) = dlsym(handle, "create");
    *(void **) (&destory_license_plate_recognition) = dlsym(handle, "destory");

    Alg_Module_License_Plate_Recognition* module = create_license_plate_recognition();

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
}

void test_video(std::string video_path)
{ 
    std::string channel_name = "test";

    //省际客运检测
    std::string bus_detection_root_dir = "/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement";
    Alg_Module_Bus_Detection* module_bus_detection = get_alg_module_bus_detection(bus_detection_root_dir);
    
    //车牌检测
    std::string license_plate_detection_root_dir = "/home/ubuntu/samples/alg_module_license_plate_detection/requirement";
    std::string license_plate_detection_dl_file = "/home/ubuntu/samples/alg_module_license_plate_detection/build/lib/bm/liblicense_plate_detection_bm_share.so";
    Alg_Module_License_Plate_Detection* module_license_plate_detection = get_alg_module_license_plate_detection(license_plate_detection_root_dir, license_plate_detection_dl_file);

    //车牌识别
    std::string license_plate_recognition_root_dir = "/home/ubuntu/samples/alg_module_license_plate_recognition/requirement";
    std::string license_plate_recognition_dl_file = "/home/ubuntu/samples/alg_module_license_plate_recognition/build/lib/bm/libalg_module_license_plate_recognition_bm_share.so";
    Alg_Module_License_Plate_Recognition* module_license_plate_recognition = get_alg_module_license_plate_recognition(license_plate_recognition_root_dir, license_plate_recognition_dl_file);

    //视频文件,结果保存
    std::string video_save_path = "/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/results/result.mp4";
    std::string event_save_path = "/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/results/event";

    //打开视频
    cv::VideoCapture video;
    video.open(video_path);
    int frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
    frame_count = frame_count < 1000 ? frame_count : frame_count;
    
    //保存视频
    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat image(height, width, CV_8UC3);
    
    cv::VideoWriter writer;
    int coder = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 25.0;
    writer.open(video_save_path, coder, fps, cv::Size(width, height), CV_8UC3);

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

        // 打印图像的尺寸
        // std::cout << "image size: " << image.rows << "x" << image.cols << std::endl;
        
        if (i%10 != 0) continue;

        std::cout << std::to_string(i+1) << "/" << std::to_string(frame_count);

        // bm_image_create(handle, image.rows, image.cols, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, &input_image);   
        // void *ptr = (void *)(&image.data[0]);                          //获取原始图片数据的指针
        // bm_image_copy_host_to_device(input_image, &ptr);               //将数据拷入device memory中

        std::shared_ptr<QyImage> input_image=from_mat(image,handle);
        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image = input_image;

        cv::Mat image_display = input_image->get_image();
        cv::imwrite("/home/ubuntu/workspace_nv_yan/alg_module_congestion_bev_detection/requirement/results/disply_v/" + channel_name + "input_image_display.jpg", image_display);


        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        //省际客车检测
        input["image"] = input_data_image;
        module_bus_detection->forward(channel_name, input, output);
        std::cout << "\t 车辆数量: " << output["vehicle"]->data.detect.size();

        // 可视化省际客车目标检测结果
        // cv::Mat image_copy = image.clone();
        // int count_ = 1;
        // for(auto &vehicle : output["vehicle"]->data.detect){
        //     // 打印box信息
        //     cout<<"-----------------"<<endl;
        //     std::cout << vehicle.x1 <<endl;
        //     std::cout << vehicle.y1 <<endl;
        //     std::cout << vehicle.x2 <<endl;
        //     std::cout << vehicle.y2 <<endl;
        //      cout<<"-----------------"<<endl;
        //     cv::rectangle(image_copy, cv::Rect(vehicle.x1, vehicle.y1, vehicle.x2-vehicle.x1, vehicle.y2-vehicle.y1), cv::Scalar(0, 255, 0), 3);
        //     cv::Rect roi(vehicle.x1, vehicle.y1, vehicle.x2-vehicle.x1, vehicle.y2-vehicle.y1);  
        //     cv::Mat dst = image(roi); 
        //     cv::imwrite("/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/vedios_1015/" + std::to_string(i) + "_" + std::to_string(count_++) + ".jpg", dst);
        // }
        // if(output["vehicle"]->data.detect.size() > 0){
        //     cv::imwrite("/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/vedios_1015/" + std::to_string(i) + ".jpg", image);
        // }
        
        
        //车牌检测
        input["vehicle"] = output["vehicle"];
        module_license_plate_detection->forward(channel_name, input, output);   
        std::cout << "\t 车牌数量: " << output["license"]->data.detect_license.size();
        // 使用检测结果的box抠出子图
        // for (auto &license : output["license"]->data.detect_license)
        // {
        //      cv::Rect roi(x1, y1, roi_width, roi_height); 
        // }
 
        //车牌识别
        input["license"] = output["license"];
        module_license_plate_recognition->forward(channel_name, input, output);

        //车辆和车牌进行整合
        output["vehicle"] = input["vehicle"];
        module_bus_detection->filter(channel_name, output, filter_output);  
        
        //可视化省际客车目标
        module_bus_detection->display(channel_name, input, filter_output);

        // bm_image_destroy(input_image);

        //保存检测视频
        auto& results = output["license"]->data.detect_license;
        std::cout << "\t 事件数量: " << results.size();
        for (int i = 0; i < results.size(); i++) 
        {   
            image = drawBox(image, results[i].x1, results[i].y1, results[i].x2, results[i].y2);
            image = drawLandms(image, 
                results[i].x1+results[i].landms_x1, 
                results[i].y1+results[i].landms_y1, 
                results[i].x1+results[i].landms_x2, 
                results[i].y1+results[i].landms_y2, 
                results[i].x1+results[i].landms_x3, 
                results[i].y1+results[i].landms_y3, 
                results[i].x1+results[i].landms_x4, 
                results[i].y1+results[i].landms_y4);
            // 根据车牌的检测结果把车牌扣出来 上面是车牌检测的四个角点的坐标
            
            // cout<<"results[i].landms_x1: "<<results[i].landms_x1<<", results[i].landms_y1: "<<results[i].landms_y1<<endl;
            // cout<<"results[i].landms_x2: "<<results[i].landms_x2<<", results[i].landms_y2: "<<results[i].landms_y2<<endl;
            // cout<<"results[i].landms_x3: "<<results[i].landms_x3<<", results[i].landms_y3: "<<results[i].landms_y3<<endl;
            // cout<<"results[i].landms_x4: "<<results[i].landms_x4<<", results[i].landms_x4: "<<results[i].landms_x4<<endl;
        }
        writer.write(image);

        //保存车辆事件检测结果
        for (auto& result : results) 
        {   
            std::string event_image_save_path = event_save_path + "/" + result.tag + "_" + std::to_string(event_count) + "_" + result.license + ".png";
            cv::imwrite(event_image_save_path,image);// result.res_images["event_image"]);
            event_count++;
        }

        std::cout << std::endl;
    }
    writer.release();

    destory_license_plate_detection(module_license_plate_detection);
    destory_license_plate_recognition(module_license_plate_recognition);
}

int main(int argc, char* argv[])
{
    if (argc == 1) {
        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios/04000004621000000.mp4");
        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios/2.mp4");

        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios_1015/1.mp4");
        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios_1015/2.mp4");
        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios_1015/3.mp4");
        test_video("/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/vedios_1015/4.mp4");
        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios_1015/5.mp4");
        // test_video("/home/ubuntu/workspace_yan/alg_module_bus_detection/requirement/vedios_1015/6.mp4");
    }
    if (argc == 2) {
        std::string video_path(argv[1]);
        test_video(video_path);
    }
}

