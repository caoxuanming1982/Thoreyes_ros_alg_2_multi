// /*
//     @Author      : shaoshengsong
//     @Date        : 2022-09-21 05:49:06
// */
#pragma once


#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


const int k_feature_dim=512;//特征维度

const std::string  k_feature_model_path ="./feature.onnx";
const std::string  k_detect_model_path ="./yolov5s.onnx";


typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;       // 检测框
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;    // 可变行数的四维矩阵，存储多个检测框
typedef Eigen::Matrix<float, 1, k_feature_dim, Eigen::RowMajor> FEATURE;  // 一个k_feature_dim维的行向量
typedef Eigen::Matrix<float, Eigen::Dynamic, k_feature_dim, Eigen::RowMajor> FEATURESS;  // 可变行数的k_feature_dim维矩阵
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;  //卡尔曼滤波器的均值（Mean）矩阵
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;  // 卡尔曼滤波器的协方差（Covariance）矩阵
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;  // 卡尔曼滤波器测量均值（Measurement Mean）矩阵
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;  // 卡尔曼滤波器测量协方差（Measurement Covariance）矩阵
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;  //由KAL_MEAN和KAL_COVA组成的pair
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>; // 由KAL_HMEAN和KAL_HCOVA组成的pair。


//main
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;   // 动态矩阵



