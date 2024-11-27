QyImage提供的api：
判断图像是否为空：
    bool is_empty()
获取图像宽度：
    int get_width()
获取图像高度：    
    int get_height()
获取图像的一份拷贝（深拷贝）：    
    std::shared_ptr<QyImage> copy()
获取图像所在设备的handle：    
    std::shared_ptr<Device_Handle> get_handle()
剪裁图像：    
    std::shared_ptr<QyImage> crop(cv::Rect box)
缩放图像：    
    std::shared_ptr<QyImage> resize(int width,int height,bool use_bilinear=true)
剪裁并缩放图像：    
    std::shared_ptr<QyImage> crop_resize(cv::Rect box,int width,int height,bool use_bilinear=true)
缩放图像，但保持原图的长宽比例，画布空余部分padding：    
    std::shared_ptr<QyImage> resize_keep_ratio(int width,int height,int value,Padding_mode mode=Padding_mode::LeftTop,bool use_bilinear=true)
剪裁并缩放图像，但保持剪裁图像的长宽比例，画布空余部分padding：        
    std::shared_ptr<QyImage> crop_resize_keep_ratio(cv::Rect box,int width,int height,int value,Padding_mode mode=Padding_mode::LeftTop,bool use_bilinear=true)
padding图像：    
    std::shared_ptr<QyImage> padding(int left,int right,int up,int down,int value)
padding图像到指定尺寸：    
    std::shared_ptr<QyImage> padding_to(int width,int height,int value,Padding_mode mode=Padding_mode::LeftTop)
仿射变换：    
    std::shared_ptr<QyImage> warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear=false)
仿射变换：        
    std::shared_ptr<QyImage> warp_affine(cv::Mat& matrix,int width,int height,bool use_bilinear=false)
批量仿射变换：        
    std::vector<std::shared_ptr<QyImage>> batch_warp_affine(std::vector<std::vector<float>> matrixes_in,int width,int height,bool use_bilinear=false)
投射变换：    
    std::shared_ptr<QyImage> warp_perspective(cv::Mat& matrix,int width,int height,bool use_bilinear=false)
投射变换：    
    std::shared_ptr<QyImage> warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear=false)
投射变换：    
    std::shared_ptr<QyImage> warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear=false)
批量投射变换：        
    std::vector<std::shared_ptr<QyImage>> batch_warp_perspective(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false)
批量投射变换：            
    std::vector<std::shared_ptr<QyImage>> batch_warp_perspective(std::vector<std::vector<cv::Point2f>>& pointss,int width,int height,bool use_bilinear=false)
图像转为rgb或bgr：    
    std::shared_ptr<QyImage> cvtcolor(bool to_rgb=false)
图像数据位格式转换：    
    std::shared_ptr<QyImage> convertTo(Data_type t)
获取cv::Mat格式的图像：            
    cv::Mat get_image()
设置当前QyImage为输入的cv::Mat格式的图像，需要指定当前是否为rgb模式，默认为bgr：                
    void set_image(cv::Mat input,bool is_rgb=false)
