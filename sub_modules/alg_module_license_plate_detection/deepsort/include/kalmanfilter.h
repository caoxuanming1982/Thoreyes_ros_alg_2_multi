#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "dataType.h"
namespace my_KalmanFilter{
    class KalmanFilter
    {
    public:
        static const double chi2inv95[10];
        KalmanFilter();
        KAL_DATA initiate(const DETECTBOX& measurement);
        void predict(KAL_MEAN& mean, KAL_COVA& covariance);
        KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);  // 投影
        KAL_DATA update(const KAL_MEAN& mean,
                        const KAL_COVA& covariance,
                        const DETECTBOX& measurement);


    // 计算门限距离（gating distance）门限距
    // 离是一个阈值，用于判断测量值与目标估计值之间的距
    // 离是否在可接受的范围内。如果测量值与目标估计值之间的
    // 距离小于门限距离，则认为测量值与目标匹配；如果距离大于
    // 门限距离，则认为测量值与目标不匹配。
        Eigen::Matrix<float, 1, -1> gating_distance(   
                const KAL_MEAN& mean,
                const KAL_COVA& covariance,
                const std::vector<DETECTBOX>& measurements,
                bool only_position = false);

    private:
        Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
        Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
        float _std_weight_position;
        float _std_weight_velocity;
    };
}

#endif // KALMANFILTER_H
