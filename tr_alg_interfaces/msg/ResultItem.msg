uint8 DATA_TYPE_DETECT=0                                        #无车牌的检测结果
uint8 DATA_TYPE_DETECT_WITH_LICENSE=1           #有车牌的检测结果
uint8 data_type                                                         #数据类型

float32 x1
float32 y1
float32 x2
float32 y2
float32 score
int32 class_id
string tag
int32 temp_idx 
bool new_obj
float32[] feature                                                               #特征向量

string str
float32[] contour
int32 region_idx

uint8[] mask_data                                                               #mask类型的结果
uint32[] mask_shape                                                     #mask的形状

ExtResultItem[] ext_result
ExtObjItem[] ext_object

float32 landms_x1
float32 landms_y1
float32 landms_x2
float32 landms_y2
float32 landms_x3
float32 landms_y3
float32 landms_x4
float32 landms_y4
string license
int32 car_idx

uint8 DETECT_STATE_SUCCESS = 0
uint8 DETECT_STATE_SMALL_REGION = 1
uint8 DETECT_STATE_LOW_SCORE = 2
uint8 state

uint8 LICENSE_COLOR_BLUE=0           
uint8 LICENSE_COLOR_GREEN=1     
uint8 LICENSE_COLOR_YELLOW=2     
uint8 LICENSE_COLOR_YELLOW_GREEN=3
uint8 LICENSE_COLOR_BLACK=4     
uint8 LICENSE_COLOR_WHITE=5     
uint8 LICENSE_COLOR_UNKNOWN=6     
uint8 license_color

uint8 LICENSE_TYPE_SINGLE=0           
uint8 LICENSE_TYPE_DOUBLE=1     
uint8 LICENSE_TYPE_UNKNOWN=2     
uint8 license_type



