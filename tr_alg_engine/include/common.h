#ifndef __COMMON_H__
#define __COMMON_H__

#include<iostream>
#include <sys/time.h>
#include<vector>
#include<sstream>

template <class Type>  std::string Num2string(Type Num);
template <class Type>  Type stringToNum(const std::string& str);


long get_time();

void supersplit(const std::string& s, std::vector<std::string>& v, const std::string& c);




#endif