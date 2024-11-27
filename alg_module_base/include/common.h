#ifndef __COMMON_H__
#define __COMMON_H__

#include<iostream>
#include <sys/time.h>
#include<vector>
#include<sstream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/types_c.h>


template <class Type>  std::string Num2string(Type Num);
template <class Type>  Type stringToNum(const std::string& str);

class Output{
public:
    std::vector<uint8_t> data;
    std::vector<int> shape;
};
#define MAX_DIMS_NUM 8
class Shape_t {
public:
  int num_dims;
  int dims[MAX_DIMS_NUM];
};

long get_time();

void supersplit(const std::string& s, std::vector<std::string>& v, const std::string& c);

std::string remove_space(const std::string& s);

void mat2Output(cv::Mat& input,Output& output);

#endif