#include "cv_lib/type_def.h"

void test_memory(std::string image_path){
    cv::Mat image_orig=cv::imread(image_path);
    std::shared_ptr<Device_Handle> handle=std::shared_ptr<Device_Handle>(get_device_handle(2));
    std::shared_ptr<QyImage> image=from_mat(image_orig,handle);
    int width=image->get_width();
    int height=image->get_height();
    std::cout<<"width:  "<<width<<std::endl;
    std::cout<<"height:  "<<height<<std::endl;
    for(int i=0;i<100000;i++){
        image=from_mat(image_orig,handle);

        std::shared_ptr<QyImage> image_t=image->resize(width/2,height/2);
        image_t=image_t->resize(width,height);
        image_t=image_t->cvtcolor(i%2==0);
        image_t=image_t->padding(100,100,100,100,0);
        image_t=image_t->crop(cv::Rect(100,100,width,height));
        if(i%1000==0){
            std::cout<<i<<std::endl;
        }
    }

}

void test_image(std::string image_path,std::string save_root){
    cv::Mat image_orig=cv::imread(image_path);
    std::shared_ptr<Device_Handle> handle=std::shared_ptr<Device_Handle>(get_device_handle(0));
    std::shared_ptr<QyImage> image=from_mat(image_orig,handle);
    int width=image->get_width();
    int height=image->get_height();
    std::cout<<"width:  "<<width<<std::endl;
    std::cout<<"height:  "<<height<<std::endl;

    cv::imwrite(save_root+"/copy.png", image->copy()->get_image());
    cv::imwrite(save_root+"/crop.png", image->crop(cv::Rect(0,0,width*0.5,height*0.5))->get_image());
    cv::imwrite(save_root+"/resize.png", image->resize(height,height)->get_image());
    cv::imwrite(save_root+"/crop_resize.png", image->crop_resize(cv::Rect(0,0,width*0.5,height*0.5),height,height)->get_image());
    cv::imwrite(save_root+"/crop_resize_keep_ratio.png", image->crop_resize_keep_ratio(cv::Rect(0,0,width*0.5,height*0.5),height,height,0)->get_image());
    cv::imwrite(save_root+"/resize_keep_ratio.png", image->resize_keep_ratio(height,height,0)->get_image());
    cv::imwrite(save_root+"/cvtcolor.png", image->cvtcolor(true)->get_image());
    cv::imwrite(save_root+"/padding.png", image->padding(10,20,30,40,0)->get_image());
    cv::imwrite(save_root+"/padding_to.png", image->padding_to(width+50,height+50,0)->get_image());
    cv::imwrite(save_root+"/padding_to_center.png", image->padding_to(width+50,height+50,0,QyImage::Padding_mode::Center)->get_image());
    
    std::vector<cv::Point2f> src_points;
    src_points.push_back(cv::Point2f(0.3*width,0.3*height));
    src_points.push_back(cv::Point2f(0.7*width,0.1*height));
    src_points.push_back(cv::Point2f(0.3*width,0.7*height));
    src_points.push_back(cv::Point2f(0.7*width,0.9*height));

    cv::Mat transform=cv::getRotationMatrix2D(cv::Point2f(width*0.3,height*0.3),30,0.8);
    transform.convertTo(transform,CV_32FC1);
    std::shared_ptr<QyImage>temp= image->warp_affine(transform,width,height);
    cv::imwrite(save_root+"/affine.png", temp->get_image());
    cv::imwrite(save_root+"/perpective.png", image->warp_perspective(src_points,width,height)->get_image());
/*
    cv::imwrite(save_root+"/add_r.png", image->operator+(cv::Scalar(0,0,50))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/sub_r.png", image->operator-(cv::Scalar(0,0,50))->convertTo(QyImage::Data_type::UInt8)->get_image());

    cv::imwrite(save_root+"/add_b.png", image->operator+(cv::Scalar(50,0,0))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/sub_b.png", image->operator-(cv::Scalar(50,0,0))->convertTo(QyImage::Data_type::UInt8)->get_image());

    cv::imwrite(save_root+"/add_rg.png", image->operator+(cv::Scalar(0,50,50))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/sub_rg.png", image->operator-(cv::Scalar(0,50,50))->convertTo(QyImage::Data_type::UInt8)->get_image());

    cv::imwrite(save_root+"/add_bg.png", image->operator+(cv::Scalar(50,50,0))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/sub_bg.png", image->operator-(cv::Scalar(50,50,0))->convertTo(QyImage::Data_type::UInt8)->get_image());

    cv::imwrite(save_root+"/add_br.png", image->operator+(cv::Scalar(50,0,50))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/sub_br.png", image->operator-(cv::Scalar(50,0,50))->convertTo(QyImage::Data_type::UInt8)->get_image());

    cv::imwrite(save_root+"/add_rgb.png", image->operator+(cv::Scalar(150,150,150))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/sub_rgb.png", image->operator-(cv::Scalar(150,150,150))->convertTo(QyImage::Data_type::UInt8)->get_image());

    cv::imwrite(save_root+"/mul.png", image->operator*(cv::Scalar(2,0.5,1))->convertTo(QyImage::Data_type::UInt8)->get_image());
    cv::imwrite(save_root+"/div.png", image->operator/(cv::Scalar(2,0.5,1))->convertTo(QyImage::Data_type::UInt8)->get_image());
    std::shared_ptr<QyImage>temp1= image->scale_add(cv::Scalar(0.5,0.5,0.5),cv::Scalar(100,100,100));
    if(temp1==nullptr){
        std::cout<<"scale add op error"<<std::endl;
    }
    cv::imwrite(save_root+"/scale_add.png", temp1->convertTo(QyImage::Data_type::UInt8)->get_image());
*/

}

int main(int argc, char *argv[])
{

#ifdef USE_BM    
    test_memory("../test_data/1.png");
#endif

#ifdef USE_CV
    test_memory("../test_data/1.png");
#endif

#ifdef USE_HW
    test_memory("../test_data/1.png");
#endif
#ifdef USE_CVCUDA
    test_memory("../test_data/1.png");
#endif
#ifdef USE_CUDA
    test_memory("../test_data/1.png");
#endif

/*
#ifdef USE_BM    
    test_image("../test_data/1.png","../test_data/bm_result/");
#endif

#ifdef USE_CV
    test_image("../test_data/1.png","../test_data/cv_result/");
#endif

#ifdef USE_HW
    test_image("../test_data/1.png","../test_data/hw_result/");
#endif
#ifdef USE_CVCUDA
    test_image("../test_data/1.png","../test_data/cvcuda_result/");
#endif
#ifdef USE_CUDA
    test_image("../test_data/1.png","../test_data/cuda_result/");
#endif
*/
}