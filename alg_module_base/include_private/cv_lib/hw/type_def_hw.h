#ifndef __CV_LIB_TYPE_DEF_HW_H__
#define __CV_LIB_TYPE_DEF_HW_H__
#include "cv_lib/type_def.h"

#define USE_HW

typedef int32_t hi_u32;
typedef void hi_void;
typedef enum {
    HI_PIXEL_FORMAT_YUV_400 = 0,              // YUV400 8bit
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1,   // YUV420SP NV12 8bit
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420 = 2,   // YUV420SP NV21 8bit
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422 = 3,   // YUV422SP 8bit
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422 = 4,   // YVU422SP 8bit
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_444 = 5,   // YUV444SP 8bit
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444 = 6,   // YVU444SP 8bit
    HI_PIXEL_FORMAT_YUYV_PACKED_422 = 7,      // YUV422 Package YUYV 8bit
    HI_PIXEL_FORMAT_UYVY_PACKED_422 = 8,      // YUV422 Package  UYVY 8bit
    HI_PIXEL_FORMAT_YVYU_PACKED_422 = 9,      // YUV422 Package  YVYU 8bit
    HI_PIXEL_FORMAT_VYUY_PACKED_422 = 10,     // YUV422 Package  VYUY 8bit
    HI_PIXEL_FORMAT_YUV_PACKED_444 = 11,      // YUV444 Package  8bit
    HI_PIXEL_FORMAT_RGB_888 = 12,             // RGB888
    HI_PIXEL_FORMAT_BGR_888 = 13,             // BGR888
    HI_PIXEL_FORMAT_ARGB_8888 = 14,           // ARGB8888       
    HI_PIXEL_FORMAT_ABGR_8888 = 15,           // ABGR8888
    HI_PIXEL_FORMAT_RGBA_8888 = 16,           // RGBA8888
    HI_PIXEL_FORMAT_BGRA_8888 = 17,            // BGRA8888
    HI_PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT = 18,    // YUV420SP 10bit
    HI_PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT = 19,    // YVU420sp 10bit
    HI_PIXEL_FORMAT_YVU_PLANAR_420 = 20,       // YVU420P 8bit
    HI_PIXEL_FORMAT_YVU_PLANAR_422 = 21,       // YVU422P 8bit  
    HI_PIXEL_FORMAT_YVU_PLANAR_444 = 22,       // YVU444P 8bit
    HI_PIXEL_FORMAT_RGB_444 = 23,              // RGB444  R:4bit G:4bit B:4bit，当前不支持该格式
    HI_PIXEL_FORMAT_BGR_444 = 24,              // BGR444  R:4bit G:4bit B:4bit，当前不支持该格式
    HI_PIXEL_FORMAT_ARGB_4444 = 25,            // ARGB4444 A:4bit R:4bit G:4bit B:4bit
    HI_PIXEL_FORMAT_ABGR_4444 = 26,            // ABGR4444 A:4bit B:4bit G:4bit R:4bit，当前不支持该格式
    HI_PIXEL_FORMAT_RGBA_4444 = 27,            // RGBA4444 R:4bit G:4bit B:4bit A:4bit，当前不支持该格式
    HI_PIXEL_FORMAT_BGRA_4444 = 28,            // BGRA4444 B:4bit G:4bit R:4bit A:4bit，当前不支持该格式
    HI_PIXEL_FORMAT_RGB_555 = 29,              // RGB555 R:5bit G:5bit B:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_BGR_555 = 30,              // BGR555 B:5bit G:5bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_RGB_565 = 31,              // RGB565 R:5bit G:6bit B:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_BGR_565 = 32,              // BGR565 B:5bit G:6bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_ARGB_1555 = 33,            // ARGB1555 A:1bit R:5bit G:6bit B:5bit
    HI_PIXEL_FORMAT_ABGR_1555 = 34,            // ABGR1555 A:1bit B:5bit G:6bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_RGBA_1555 = 35,            // RGBA1555 A:1bit B:5bit G:6bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_BGRA_1555 = 36,            // BGRA1555 A:1bit B:5bit G:6bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_ARGB_8565 = 37,            // ARGB8565 A:8bit R:5bit G:6bit B:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_ABGR_8565 = 38,            // ABGR8565 A:8bit B:5bit G:6bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_RGBA_8565 = 39,            // RGBA8565 A:8bit R:5bit G:6bit B:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_BGRA_8565 = 40,            // BGRA8565 A:8bit B:5bit G:6bit R:5bit，当前不支持该格式
    HI_PIXEL_FORMAT_ARGB_CLUT2 = 41,           // ARGB Color Lookup Table 2bit     
    HI_PIXEL_FORMAT_ARGB_CLUT4 = 42,           // ARGB Color Lookup Table 4bit

    HI_PIXEL_FORMAT_RGB_BAYER_8BPP = 50,       
    HI_PIXEL_FORMAT_RGB_BAYER_10BPP = 51,      
    HI_PIXEL_FORMAT_RGB_BAYER_12BPP = 52,      
    HI_PIXEL_FORMAT_RGB_BAYER_14BPP = 53,      
    HI_PIXEL_FORMAT_RGB_BAYER_16BPP = 54,      // RGB Bayer 16bit，Bayer图像，当前不支持该格式
    HI_PIXEL_FORMAT_YUV_PLANAR_420 = 55,       // YUV420P 8bit
    HI_PIXEL_FORMAT_YUV_PLANAR_422 = 56,       // YUV422P 8bit
    HI_PIXEL_FORMAT_YUV_PLANAR_444 = 57,       // YUV444P 8bit
    HI_PIXEL_FORMAT_YVU_PACKED_444 = 58,       // YVU444 Package 8bit
    HI_PIXEL_FORMAT_XYUV_PACKED_444 = 59,      // AYUV444 Package 8bit
    HI_PIXEL_FORMAT_XYVU_PACKED_444 = 60,      // AYVU444 Package 8bit
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_411 = 61,   // YUV411SP 8bit
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_411 = 62,   // YVU411SP 8bit
    HI_PIXEL_FORMAT_YUV_PLANAR_411 = 63,       // YUV411P 8bit
    HI_PIXEL_FORMAT_YVU_PLANAR_411 = 64,       // YVU411P 8bit
    HI_PIXEL_FORMAT_YUV_PLANAR_440 = 65,       // YUV440P 8bit
    HI_PIXEL_FORMAT_YVU_PLANAR_440 = 66,       // YVU440P 8bit

    HI_PIXEL_FORMAT_RGB_888_PLANAR = 69,       // RGB888 Planar
    HI_PIXEL_FORMAT_BGR_888_PLANAR = 70,       // BGR888 Planar
    HI_PIXEL_FORMAT_HSV_888_PACKAGE = 71,      // HSV Package，HSV图像package格式，当前不支持该格式
    HI_PIXEL_FORMAT_HSV_888_PLANAR = 72,       // HSV Planar，HSV图像Planar格式，当前不支持该格式
    HI_PIXEL_FORMAT_LAB_888_PACKAGE = 73,      // LAB Package，LAB图像package格式，当前不支持该格式
    HI_PIXEL_FORMAT_LAB_888_PLANAR = 74,       // LAB Planar，LAB图像Planar格式，当前不支持该格式
    HI_PIXEL_FORMAT_S8C1 = 75,                 // Signed 8bit for 1pixel 1Channel，每个像素用1个8bit有符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_S8C2_PACKAGE = 76,         // Signed 8bit for 1pixel 2Channel Package，每个像素用2个8bit有符号数表示的双通道图像Package格式，当前不支持该格式
    HI_PIXEL_FORMAT_S8C2_PLANAR = 77,          // Signed 8bit for 1pixel 2Channel Planar，每个像素用2个8bit有符号数据表的双通道图像Planar格式，当前不支持该格式
    HI_PIXEL_FORMAT_S16C1 = 78,                // Signed 16bit 1pixel 1Channel，每个像素用1个16bit有符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_U8C1 = 79,                 // Unsigned 8bit 1pixel 1Channel，每个像素用1个8bit无符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_U16C1 = 80,                // Unsigned 16bit 1pixel 1Channel，每个像素用1个16bit无符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_S32C1 = 81,                // Signed 32bit 1pixel 1Channel，每个像素用1个32bit有符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_U32C1 = 82,                // Unsigned 32bit 1pixel 1Channel，每个像素用1个32bit无符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_U64C1 = 83,                // Unsigned 64bit 1pixel 1Channel，每个像素用1个64bit无符号数据表示的单通道图像，当前不支持该格式
    HI_PIXEL_FORMAT_S64C1 = 84,                // Signed 64bit 1pixel 1Channel，每个像素用1个64bit有符号数据表示的单通道图像，当前不支持该格式


    HI_PIXEL_FORMAT_RGB_888_INT8 = 110,        // RGB888 Package 每个像素的单分量占用1个8bit有符号数
    HI_PIXEL_FORMAT_BGR_888_INT8 = 111,        // BGR888 Package 每个像素的单分量占用1个8bit有符号数
    HI_PIXEL_FORMAT_RGB_888_INT16 = 112,       // RGB888 Package 每个像素的单分量占用1个16bit有符号数
    HI_PIXEL_FORMAT_BGR_888_INT16 = 113,       // BGR888 Package 每个像素的单分量占用1个16bit有符号数
    HI_PIXEL_FORMAT_RGB_888_INT32 = 114,       // RGB888 Package 每个像素的单分量占用1个32bit有符号数
    HI_PIXEL_FORMAT_BGR_888_INT32 = 115,       // BGR888 Package 每个像素的单分量占用1个32bit有符号数
    HI_PIXEL_FORMAT_RGB_888_UINT16 = 116,      // RGB888 Package 每个像素的单分量占用1个16bit无符号数
    HI_PIXEL_FORMAT_BGR_888_UINT16 = 117,      // BGR888 Package 每个像素的单分量占用1个16bit无符号数
    HI_PIXEL_FORMAT_RGB_888_UINT32 = 118,      // RGB888 Package 每个像素的单分量占用1个32bit无符号数
    HI_PIXEL_FORMAT_BGR_888_UINT32 = 119,      // BGR888 Package 每个像素的单分量占用1个32bit无符号数
    HI_PIXEL_FORMAT_RGB_888_PLANAR_INT8  = 120,// RGB888 Planar 每个像素的单分量占用1个8bit有符号数
    HI_PIXEL_FORMAT_BGR_888_PLANAR_INT8  = 121,// BGR888 Planar 每个像素的单分量占用1个8bit有符号数
    HI_PIXEL_FORMAT_RGB_888_PLANAR_INT16 = 122,// RGB888 Planar 每个像素的单分量占用1个16bit有符号数
    HI_PIXEL_FORMAT_BGR_888_PLANAR_INT16 = 123,// BGR888 Planar 每个像素的单分量占用1个16bit有符号数
    HI_PIXEL_FORMAT_RGB_888_PLANAR_INT32 = 124,// RGB888 Planar 每个像素的单分量占用1个32bit有符号数
    HI_PIXEL_FORMAT_BGR_888_PLANAR_INT32 = 125,// BGR888 Planar 每个像素的单分量占用1个32bit有符号数
    HI_PIXEL_FORMAT_RGB_888_PLANAR_UINT16 = 126,// RGB888 Planar 每个像素的单分量占用1个16bit无符号数
    HI_PIXEL_FORMAT_BGR_888_PLANAR_UINT16 = 127,// BGR888 Planar 每个像素的单分量占用1个16bit无符号数
    HI_PIXEL_FORMAT_RGB_888_PLANAR_UINT32 = 128,// RGB888 Planar 每个像素的单分量占用1个32bit无符号数
    HI_PIXEL_FORMAT_BGR_888_PLANAR_UINT32 = 129,// BGR888 Planar 每个像素的单分量占用1个32bit无符号数
    HI_PIXEL_FORMAT_YUV400_UINT16 = 130,       // YUV400 Package 每个像素的单分量占用1个16bit无符号数
    HI_PIXEL_FORMAT_YUV400_UINT32 = 131,       // YUV400 Package 每个像素的单分量占用1个32bit无符号数
    HI_PIXEL_FORMAT_YUV400_UINT64 = 132,       // YUV400 Package 每个像素的单分量占用1个64bit无符号数
    HI_PIXEL_FORMAT_YUV400_INT8   = 133,       // YUV400 Package 每个像素的单分量占用1个8bit有符号数
    HI_PIXEL_FORMAT_YUV400_INT16  = 134,       // YUV400 Package 每个像素的单分量占用1个16bit有符号数
    HI_PIXEL_FORMAT_YUV400_INT32  = 135,       // YUV400 Package 每个像素的单分量占用1个32bit有符号数
    HI_PIXEL_FORMAT_YUV400_INT64  = 136,       // YUV400 Package 每个像素的单分量占用1个64bit有符号数
    HI_PIXEL_FORMAT_YUV400_FP16 = 137,         // YUV400 Package 每个像素用1个float16数据表示
    HI_PIXEL_FORMAT_YUV400_FP32 = 138,         // YUV400 Package 每个像素用1个float32数据表示
    HI_PIXEL_FORMAT_YUV400_FP64 = 139,         // YUV400 Package 每个像素用1个float64数据表示
    HI_PIXEL_FORMAT_YUV400_BF16 = 140,         // YUV400 Package 每个像素用1个BFloat16数据表示

    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_440 = 1000, // YUV440SP 8bit
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_440 = 1001, // YVU440SP 8bit
    HI_PIXEL_FORMAT_FLOAT32 = 1002,            // Float 32bit for 1pixel，每个像素用1个float32数据表示，当前不支持该格式
    HI_PIXEL_FORMAT_BUTT = 1003,

    HI_PIXEL_FORMAT_RGB_888_PLANAR_FP16 = 1004,// RGB888 Planar 每个像素用1个float16数据表示
    HI_PIXEL_FORMAT_BGR_888_PLANAR_FP16 = 1005,// BGR888 Planar 每个像素用1个float16数据表示
    HI_PIXEL_FORMAT_RGB_888_PLANAR_FP32 = 1006,// RGB888 Planar 每个像素用1个float32数据表示
    HI_PIXEL_FORMAT_BGR_888_PLANAR_FP32 = 1007,// BGR888 Planar 每个像素用1个float32数据表示
    HI_PIXEL_FORMAT_RGB_888_PLANAR_BF16 = 1008,// RGB888 Planar 每个像素用1个BFloat16数据表示
    HI_PIXEL_FORMAT_BGR_888_PLANAR_BF16 = 1009,// BGR888 Planar 每个像素用1个BFloat16数据表示
    HI_PIXEL_FORMAT_RGB_888_FP16 = 1010,       // RGB888 Package，每个像素用1个float16数据表示
    HI_PIXEL_FORMAT_BGR_888_FP16 = 1011,       // BGR888 Package，每个像素用1个float16数据表示
    HI_PIXEL_FORMAT_RGB_888_FP32 = 1012,       // RGB888 Package，每个像素用1个float32数据表示
    HI_PIXEL_FORMAT_BGR_888_FP32 = 1013,       // BGR888 Package，每个像素用1个float32数据表示
    HI_PIXEL_FORMAT_RGB_888_BF16 = 1014,       // RGB888 Package 每个像素用1个BFloat16数据表示
    HI_PIXEL_FORMAT_BGR_888_BF16 = 1015,       // BGR888 Package 每个像素用1个BFloat16数据表示

    HI_PIXEL_FORMAT_UNKNOWN = 10000
} hi_pixel_format;

typedef struct {
    hi_void* picture_address;
    hi_u32 picture_buffer_size;
    hi_u32 picture_width;
    hi_u32 picture_height;
    hi_u32 picture_width_stride;
    hi_u32 picture_height_stride;
    hi_pixel_format picture_format;
} hi_vpc_pic_info;

class HW_DVPP_Chn{
    hi_vpc_chn chnId;
    hi_vpc_chn_attr stChnAttr;
    bool inited=false;
public:
    HW_DVPP_Chn(){

    };   
    void init(){
        if(inited==false){
            int32_t ret=hi_mpi_vpc_sys_create_chn(&chnId, &stChnAttr);
            inited=true;

        }
    };
    ~HW_DVPP_Chn(){
        inited=false;
        int32_t ret = hi_mpi_vpc_destroy_chn(chnId);
    };

    hi_vpc_chn get_chn(){
        return chnId;
    };
    hi_vpc_chn_attr get_chn_attr(){
        return stChnAttr;
    };
};

struct Image_hw_
{							
	int device_idx=0;																//计算卡的图像数据
    aclrtStream device_stream;    
    std::shared_ptr<HW_DVPP_Chn> chn;
    hi_vpc_pic_info image;
	~Image_hw_(){
        if(image.picture_address!=nullptr){
            int32_t ret = hi_mpi_dvpp_free(image.picture_address);
            image.picture_address=nullptr;
        }

    };
};


class QyImage_hw:public QyImage{

public:
    Image_hw_ data;

    QyImage_hw(std::shared_ptr<Device_Handle> handle):QyImage(QyImage::Type::Image_bm_t,handle){
                
    };
    virtual ~QyImage_hw(){};
    virtual int get_width();
    virtual int get_height();
    virtual bool is_empty();

    virtual std::shared_ptr<QyImage> copy();
    virtual std::shared_ptr<QyImage> crop(cv::Rect box);
    virtual std::shared_ptr<QyImage> resize(int width,int height,bool use_bilinear=true);
    virtual std::shared_ptr<QyImage> crop_resize(cv::Rect box,int width,int height,bool use_bilinear=true);

    virtual std::shared_ptr<QyImage> padding(int left,int right,int up,int down,int value);

    virtual std::shared_ptr<QyImage> warp_affine(std::vector<float>& matrix,int width,int height,bool use_bilinear=false);

    virtual std::shared_ptr<QyImage> warp_perspective(std::vector<float>& matrix,int width,int height,bool use_bilinear=false);
    virtual std::shared_ptr<QyImage> warp_perspective(std::vector<cv::Point2f>& points,int width,int height,bool use_bilinear=false);

    virtual std::vector<std::shared_ptr<QyImage>> batch_warp_affine(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false);  //批量仿射变换，部分硬件下，效率比循环进行单个仿射变换效率高

    virtual std::vector<std::shared_ptr<QyImage>> batch_warp_perspective(std::vector<std::vector<float>>& matrixes_in,int width,int height,bool use_bilinear=false);            //批量透射变换，部分硬件下，效率比循环进行单个仿射变换效率高
    virtual std::vector<std::shared_ptr<QyImage>> batch_warp_perspective(std::vector<std::vector<cv::Point2f>>& pointss,int width,int height,bool use_bilinear=false);          //批量透射变换，部分硬件下，效率比循环进行单个仿射变换效率高

    virtual std::shared_ptr<QyImage> cvtcolor(bool to_rgb=false);

    virtual std::shared_ptr<QyImage> convertTo(QyImage::Data_type t);

    virtual cv::Mat get_image();
    virtual void set_image(cv::Mat input,bool is_rgb=false);


};

std::shared_ptr<HW_DVPP_Chn> get_dvpp_chn();


#endif