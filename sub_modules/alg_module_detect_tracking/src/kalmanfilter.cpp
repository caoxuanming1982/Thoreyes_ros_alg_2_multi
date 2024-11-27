#include "kalmanfilter.h"
#include <eigen3/Eigen/Cholesky>
using namespace my_KalmanFilter;

#define INIT_X 2.0
#define INIT_Y 2.0
#define INIT_R 2.0
#define INIT_H 2.0
#define PRED_X 2.0
#define PRED_Y 2.0
#define PRED_R 1.0
#define PRED_H 2.0


//sisyphus
const double KalmanFilter::chi2inv95[10] = { // chi2inv95 是一个包含10个元素的常量数组
//，每个元素都是一个 double 类型的值。这些值表示卡方分布的临界值，当统计量的值超过这些临界值时，可以以95%的置信水平拒绝原假设。
    0,
    3.8415,
    5.9915,
    7.8147,
    9.4877,
    11.070,
    12.592,
    14.067,
    15.507,
    16.919};
KalmanFilter::KalmanFilter()
{
    int ndim = 4;  // 表示目标状态的维度
    double dt = 1.;  // 表示时间步长。

    _motion_mat = Eigen::MatrixXf::Identity(8, 8); // 单位矩阵
    for (int i = 0; i < ndim; i++)  // _motion_mat就是Fk
    {
        _motion_mat(i, ndim + i) = dt;
    }
    _update_mat = Eigen::MatrixXf::Identity(4, 8); // 就是H, 状态变量（观测）的转换矩阵，表示将状态和观测连接起来的关系

    this->_std_weight_position = 1. / 20;
    this->_std_weight_velocity = 1. / 160; // 为位置和速度分配不同的标准权重，可以控制在状态估计和预测中对位置和速度的重要性程度。
}

KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
{
    DETECTBOX mean_pos = measurement; // 表示目标的初始位置和初始速度。
    DETECTBOX mean_vel;
    for (int i = 0; i < 4; i++)
        mean_vel(i) = 0;

    KAL_MEAN mean; // 初始化卡尔曼滤波器的初始均值（x)
    for (int i = 0; i < 8; i++)
    {
        if (i < 4) 
            mean(i) = mean_pos(i);
        else if (i >= 4 && i < 8)
            mean(i) = mean_vel(i - 4);
        else
            mean(i) = 1e-8;
    }

    KAL_MEAN std;  // 初始化卡尔曼滤波器的标准差
    std(0) = INIT_X * _std_weight_position * measurement[0];
    std(1) = INIT_Y * _std_weight_position * measurement[1];
    std(2) = INIT_R * _std_weight_position * measurement[2];
    std(3) = INIT_H * _std_weight_position * measurement[3];
    std(4) = 10 * _std_weight_velocity * measurement[0];
    std(5) = 10 * _std_weight_velocity * measurement[1];
    std(6) = 10 * _std_weight_velocity * measurement[2];
    std(7) = 10 * _std_weight_velocity * measurement[3];

    KAL_MEAN tmp = std.array().square();
    KAL_COVA var = tmp.asDiagonal();  // 使用std来计算var(该协方差矩阵是对角矩阵，代表各个变量之间没有相关性)
    return std::make_pair(mean, var);
}

void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
{
    // revise the data;
    DETECTBOX std_pos;
    DETECTBOX std_vel;
    std_pos << _std_weight_position * mean(0) * PRED_X, _std_weight_position * mean(1) * PRED_Y, 1 * PRED_R, _std_weight_position * mean(3) * PRED_H;
    std_vel << _std_weight_velocity * mean(0), _std_weight_velocity * mean(1), 1e-5, _std_weight_velocity * mean(3);

    KAL_MEAN tmp;
    tmp.block<1, 4>(0, 0) = std_pos;
    tmp.block<1, 4>(0, 4) = std_vel;
    tmp = tmp.array().square();
    KAL_COVA motion_cov = tmp.asDiagonal();
    KAL_MEAN mean1 = this->_motion_mat * mean.transpose(); // 这个是预测x，没有考虑噪音，xk = Axk-1
    KAL_COVA covariance1 = this->_motion_mat * covariance * (_motion_mat.transpose()); // 预测协方差矩阵，也没有考虑噪音，Pk = A.Pk-1.AT
    covariance1 += motion_cov;

    mean = mean1;
    covariance = covariance1;
}

KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance) // 状态的均值和协方差映射到测量空间
{
    DETECTBOX std;
    std << _std_weight_position * mean(0), _std_weight_position * mean(1), 1e-1, _std_weight_position * mean(3);
    
    KAL_HMEAN mean1 = _update_mat * mean.transpose();
    KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    diag = diag.array().square().matrix();
    covariance1 += diag;
    //    covariance1.diagonal() << diag;
    return std::make_pair(mean1, covariance1);
}

KAL_DATA
KalmanFilter::update(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const DETECTBOX &measurement)
{
    KAL_HDATA pa = project(mean, covariance);
    KAL_HMEAN projected_mean = pa.first;
    KAL_HCOVA projected_cov = pa.second;

    // chol_factor, lower =
    // scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
    // kalmain_gain =
    // scipy.linalg.cho_solve((cho_factor, lower),
    // np.dot(covariance, self._upadte_mat.T).T,
    // check_finite=False).T
    Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean;                // eg.1x4
    auto tmp = innovation * (kalman_gain.transpose());
    KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
    KAL_COVA new_covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
    return std::make_pair(new_mean, new_covariance);
}

Eigen::Matrix<float, 1, -1>
KalmanFilter::gating_distance( 
    // 卡尔曼滤波器中的门限距离计算函数。它通过将测量值与状态估计值映射到测量空间，并
    //计算马氏距离的平方，来评估测量值与状态估计值之间的差异。这样，我们可以根据门限距离来决定是否接受或拒绝测量值，以提高状态估计的准确性。
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const std::vector<DETECTBOX> &measurements,
    bool only_position)
{
    KAL_HDATA pa = this->project(mean, covariance);
    if (only_position)
    {
        printf("not implement!");
        exit(0);
    }
    KAL_HMEAN mean1 = pa.first;
    KAL_HCOVA covariance1 = pa.second;

    //    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
    DETECTBOXSS d(measurements.size(), 4);
    int pos = 0;
    for (DETECTBOX box : measurements)
    {
        d.row(pos++) = box - mean1;
    }
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
    Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
    auto zz = ((z.array()) * (z.array())).matrix();
    auto square_maha = zz.colwise().sum();
    return square_maha;
}
