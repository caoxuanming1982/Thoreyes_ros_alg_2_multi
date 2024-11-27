#include "common.h"
#include <iostream>

#include <sstream>  
#include <string> 

template <class Type>  
std::string Num2string(Type Num)
{
	std::ostringstream oss;
	oss<<Num;
	std::string str(oss.str());
	return str;
};
template <class Type>  
Type stringToNum(const std::string& str)  
{  
    std::istringstream iss(str);  
    Type num;  
    iss >> num;  
    return num;      
} 

template std::string Num2string<int>(int Num);
template std::string Num2string<float>(float Num );
template std::string Num2string<double>(double Num);
template std::string Num2string<long>(long Num);

template int stringToNum<int>(const std::string& str);
template float stringToNum<float>(const std::string& str);
template long stringToNum<long>(const std::string& str);
template double stringToNum<double>(const std::string& str);


long get_time(){
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec*1000+tv.tv_usec/1000;
};

void supersplit(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    size_t len = s.length();
    pos2 = s.find(c);
    pos1 = 0;
//    cout<<s<<endl;
    while(std::string::npos != pos2)
    {
        v.emplace_back(s.substr(pos1, pos2-pos1));
 
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != len)
        v.emplace_back(s.substr(pos1));
};

void mat2Output(cv::Mat& input,Output& output){
    output.shape.clear();
    output.data.clear();
    output.shape.push_back(1);
    cv::Mat input_t;
    if(input.isContinuous()==false){

        input_t=input.clone();
    }
    else{
        input_t=input;
    }
    output.shape.push_back(input_t.rows);
    output.shape.push_back(input_t.cols);
    output.shape.push_back(input_t.channels());
    output.data.resize(input_t.elemSize()*input_t.rows*input_t.cols);
    memcpy(output.data.data(),input_t.data,output.data.size());
};

bool is_whitespace(char c) {
    return std::isspace(static_cast<unsigned char>(c));
}

std::string remove_space(const std::string& s){
    std::string str=s;
    str.erase(std::remove_if(str.begin(),str.end(),is_whitespace),str.end());
    return str;
};
