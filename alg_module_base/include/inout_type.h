#ifndef __INOUT_TYPE_H__
#define __INOUT_TYPE_H__

#include <iostream>

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>


#include <thread>
#include <mutex>
#include <unistd.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <shared_mutex>
#include "cv_lib/type_def.h"
using string = std::string;

struct Value
{
	enum Type
	{
		Long,
		Int,
		Float,
		Double,
		String
	};
	Type data_type;
	union U
	{
		long long long_value;
		int int_value;
		float float_value;
		double double_value;
		std::string string_value;
		std::string message;
		U(Type t)
		{
			if (t == Type::Long)
			{
				long_value=0;
			}
			else if (t == Type::Int)
			{
				int_value=0;
			}
			else if (t == Type::Float)
			{
				float_value=0;
			}
			else if (t == Type::Double)
			{
				double_value=0;
			}
			else if (t == Type::String)
			{
				new (&string_value) std::string();
			}
			else
			{
				new (&message) std::string("error type");
			}
		};
		~U(){

		};


	}data;
	Value(Type t=Type::Int) : data(t){
		data_type=t;
	};
	~Value()
	{

		if (data_type == Type::String)
		{
			this->data.string_value.~string();
		}
		else if (data_type == Type::Int)
		{
		}
		else if (data_type == Type::Long)
		{
		}
		else if (data_type == Type::Float)
		{
		}
		else if (data_type == Type::Double)
		{
		}
		else
		{
			this->data.message.~string();
		}
	};

};

struct Value_Array{
	std::vector<Value> values;
};

struct Value_Dict{
	std::map<std::string,Value> values;
};


struct Ext_Result{
	float score;
	int class_id;
	std::string tag;
};

struct Result_Detect							
{																							//检测和分割结果
	float x1, x2, y1, y2;
	float score;
	int class_id;
	std::string tag;
	int temp_idx = -1;
	bool new_obj;
	std::vector<float> feature;

	std::vector<std::pair<float, float>> contour;
	int region_idx;

	std::vector<uint8_t> mask_data;
	std::vector<uint32_t> mask_shape;

	std::map<std::string,cv::Mat> res_images;


	std::map<std::string,Ext_Result> ext_result;

};
struct Result_Detect_license : Result_Detect
{																							//含车牌的检测结果
	enum Detect_state
	{
		SUCCESS,
		SMALL_REGION,
		LOW_SCORE
	};
	enum License_Color{
        Blue,
        Green,
        Yellow,
        Yellow_Green,
        Black,
        White,
        Color_UNKNOWN
	};
	enum License_Type{
        Single,
        Double,
        Type_UNKNOWN
	};

	float landms_x1;
	float landms_y1;
	float landms_x2;
	float landms_y2;
	float landms_x3;
	float landms_y3;
	float landms_x4;
	float landms_y4;
	std::string license = "";
	int car_idx=-1;

	Detect_state state=Detect_state::SUCCESS;
	License_Color license_color=License_Color::Color_UNKNOWN;
	License_Type license_type=License_Type::Type_UNKNOWN;
};
struct PointItem{
	float x=0,y=0,z=0;
	float doppler=0;
	float intensity=0;
	~PointItem(){};
};


struct PointCloud
{																							//点云的数据（后续会有修改）
	std::vector<PointItem> data;
	~PointCloud(){};
};

struct ObjectItem{

	float x=0,y=0,z=0;
	float vx=0,vy=0,vz=0;
	uint32_t track_id=0;
	float yaw=0;
	float ext_0;
	float ext_1;
	float ext_2;
	float ext_3;
	float ext_4;
	float ext_5;
	float ext_6;
	float ext_7;
	float ext_8;
	~ObjectItem(){};
	std::string to_string(){
		std::string res="";
		res+="\ttrack_id:"+std::to_string(track_id)+"\n";
		res+="\tyaw:"+std::to_string(yaw)+"\n";
		res+="\tx:"+std::to_string(x)+"\t";
		res+="\ty:"+std::to_string(y)+"\t";
		res+="\tz:"+std::to_string(z)+"\n";
		res+="\tvx:"+std::to_string(vx)+"\t";
		res+="\tvy:"+std::to_string(vy)+"\t";
		res+="\tvz:"+std::to_string(vz)+"\n";
		res+="\text:";
			res+=std::to_string(ext_0)+",";
			res+=std::to_string(ext_1)+",";
			res+=std::to_string(ext_2)+",";
			res+=std::to_string(ext_3)+",";
			res+=std::to_string(ext_4)+",";
			res+=std::to_string(ext_5)+",";
			res+=std::to_string(ext_6)+",";
			res+=std::to_string(ext_7)+",";
			res+=std::to_string(ext_8)+",";
		res+="\n";
		return res;
	}
};

struct ObjectCloud
{																							//点云的数据（后续会有修改）
	std::vector<ObjectItem> data;
	~ObjectCloud(){};
	std::string to_string(){
		std::string res="";
		for(auto item:data){
			res+=item.to_string();
		}
		return res;
	}
};

struct Feature
{																							//特征图的数据
	std::vector<float> data;
	std::vector<int> shape;
	~Feature(){};
};

struct InputOutput
{
	enum Type
	{
		Result_Detect_t,
		Result_Detect_license_t,
		Image_t,
		Images_t,
		PointCloud_t,
		ObjectCloud_t,
		Feature_t,
		Value_t,
		Value_Array_t,
		Value_Dict_t,
		UNKNOWN
	};
	static Type string2type(std::string type_str){
		std::transform(type_str.begin(), type_str.end(), type_str.begin(), [](char& c){
    	    return std::tolower(c);
	    });
		if(type_str=="result_detect_t"||type_str=="result_detect"){
			return Type::Result_Detect_t;
		}
		else if(type_str=="result_detect_license_t"||type_str=="result_detect_icense"||type_str=="result_detect_license"){
			return Type::Result_Detect_license_t;
		}
		else if(type_str=="image_bm_t"||type_str=="image_bm"||
			type_str=="image_cv_t"||type_str=="image_cv"||
			type_str=="image_cv_gpu_t"||type_str=="image_cv_gpu"||
			type_str=="image_tensor_t"||type_str=="image_tensor"){
			return Type::Image_t;
		}
		else if(type_str=="pointcloud_t"||type_str=="pointcloud"){
			return Type::PointCloud_t;
		}
		else if(type_str=="objectcloud_t"||type_str=="objectcloud"){
			return Type::ObjectCloud_t;
		}
		else if(type_str=="feature_t"||type_str=="feature"){
			return Type::Feature_t;
		}
		else if(type_str=="images"||type_str=="images_t"){
			return Type::Images_t;
		}
		else if(type_str=="value_t"||type_str=="value"){
			return Type::Value_t;
		}
		else if(type_str=="value_array_t"||type_str=="value_array"){
			return Type::Value_Array_t;
		}
		else if(type_str=="value_dict_t"||type_str=="value_dict"){
			return Type::Value_Dict_t;
		}
		else{
			return Type::UNKNOWN;
		}

	};
	static std::string type2string(Type type_t){
		switch(type_t){
		case Type::Result_Detect_t: return "Result_Detect_t";
		case Type::Result_Detect_license_t: return "Result_Detect_license_t";
		case Type::Image_t: return "Image_t";
		case Type::Images_t: return "Images_t";
		case Type::PointCloud_t: return "PointCloud_t";
		case Type::ObjectCloud_t: return "ObjectCloud_t";
		case Type::Feature_t: return "Feature_t";
		case Type::Value_t: return "Value_t";
		case Type::Value_Array_t: return "Value_Array_t";
		case Type::Value_Dict_t: return "Value_Dict_t";
		case Type::UNKNOWN: return "UNKNOWN";
		}
		return "";

	};

	Type data_type;
	union U
	{
		std::vector<Result_Detect> detect;
		std::vector<Result_Detect_license> detect_license;

		std::shared_ptr<QyImage> image;

		std::vector<std::shared_ptr<QyImage>> images;

		PointCloud pointcloud;
		ObjectCloud objectcloud;
		Feature feature;
		std::string message;
		Value value;
		Value_Array value_array;
		Value_Dict value_dict;

		U(Type t,Value::Type t1=Value::Type::Int)
		{
			if (t == Type::Result_Detect_t)
			{
				new (&detect) std::vector<Result_Detect>();
			}
			else if (t == Type::Result_Detect_license_t)
			{
				new (&detect_license) std::vector<Result_Detect_license>();
			}
			else if (t == Type::Image_t)
			{
				new (&image) std::shared_ptr<QyImage>();
			}
			else if (t == Type::PointCloud_t)
			{
				new (&pointcloud) PointCloud();
			}
			else if (t == Type::ObjectCloud_t)
			{
				new (&objectcloud) ObjectCloud();
			}
			else if (t == Type::Feature_t)
			{
				new (&feature) Feature();
			}
			else if (t == Type::Images_t)
			{
				new (&images) std::vector<std::shared_ptr<QyImage>>();
			}
			else if (t == Type::Value_t)
			{
				new (&value) Value(t1);
			}
			else if (t == Type::Value_Array_t)
			{
				new (&value_array) Value_Array();
			}
			else if (t == Type::Value_Dict_t)
			{
				new (&value_dict) Value_Dict();
			}
			else
			{
				new (&message) std::string("error type");
			}
		};
		~U(){

		};

	} data;
	InputOutput(Type t,Value::Type t1=Value::Type::Int) : data(t,t1){
		data_type=t;
						  };
	~InputOutput()
	{

		if (data_type == Type::Result_Detect_t)
		{
			this->data.detect.~vector();
		}
		else if (data_type == Type::Result_Detect_license_t)
		{
			this->data.detect_license.~vector();
		}
		else if (data_type == Type::Image_t)
		{
			this->data.image.reset();
		}
		else if (data_type == Type::PointCloud_t)
		{
			this->data.pointcloud.~PointCloud();
		}
		else if (data_type == Type::ObjectCloud_t)
		{
			this->data.objectcloud.~ObjectCloud();
		}
		else if (data_type == Type::Feature_t)
		{
			this->data.feature.~Feature();
		}
		else if (data_type == Type::Images_t)
		{
			this->data.images.~vector();
		}
		else if (data_type == Type::Value_t)
		{
			this->data.value.~Value();
		}
		else if (data_type == Type::Value_Array_t)
		{
			this->data.value_array.~Value_Array();
		}
		else if (data_type == Type::Value_Dict_t)
		{
			this->data.value_dict.~Value_Dict();
		}
		else
		{
			this->data.message.~string();
		}
	};
};

struct Input_cfg_item{

public:
	InputOutput::Type data_type=InputOutput::Type::UNKNOWN;
	std::string required_from_module="";
	std::string required_from_module_output_name="";
	std::string to_string();

	Input_cfg_item();
	~Input_cfg_item();
};

struct InputOutput_cfg{
	std::map<std::string,Input_cfg_item> input_cfgs;
	std::map<std::string,InputOutput::Type> output_cfgs;
	std::string module_name;

	std::string to_string(){
		std::string res="input_cfg:\n";
		for(auto iter=input_cfgs.begin();iter!=input_cfgs.end();iter++){
			res+="\t"+iter->first+" \n\t\t "+iter->second.to_string()+"\n";
		}

		res+="output_cfg:\n";
		for(auto iter=output_cfgs.begin();iter!=output_cfgs.end();iter++){
			res+="\t"+iter->first+" \n\t\t "+"type:"+InputOutput::type2string(iter->second)+"\n";
		}
		return res;
	};

	~InputOutput_cfg(){};
};

#endif