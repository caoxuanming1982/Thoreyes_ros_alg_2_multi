#include "tracker.h"
#include "nn_matching.h"
#include "model.h"
#include "linear_assignment.h"
using namespace std;

//#define MY_inner_DEBUG
#ifdef MY_inner_DEBUG
#include <string>
#include <iostream>
#endif

tracker::tracker(/*NearNeighborDisMetric *metric,*/
                 float max_cosine_distance, int nn_budget,
                 float max_iou_distance, int max_age, int n_init)
{
    this->metric = new NearNeighborDisMetric(
        NearNeighborDisMetric::METRIC_TYPE::cosine,
        max_cosine_distance, nn_budget);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;

    this->kf = new KalmanFilter();
    this->tracks.clear();
    this->_next_idx = 1;
}

void tracker::predict()
{   // 使用 kf 预测所有目标框的位置

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (Track &track : tracks)
    {
        track.predit(kf);
    }
}

void tracker::update(const DETECTIONS &detections)
{
    TRACHER_MATCHD res;
    // 调用了一个名为_match的函数，将检测结果detections与跟踪器中的轨迹进行匹配，得到匹配结果res
    _match(detections, res);

    vector<MATCH_DATA> &matches = res.matches;

    // 这段代码用于更新已匹配的轨迹
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (MATCH_DATA &data : matches)
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }
    // 用于标记未匹配的轨迹为miss状态
    vector<int> &unmatched_tracks = res.unmatched_tracks;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int &track_idx : unmatched_tracks)
    {
        this->tracks[track_idx].mark_missed();
    }
    // 用于初始化未匹配的检测结果为新的轨迹
    vector<int> &unmatched_detections = res.unmatched_detections;
    for (int &detection_idx : unmatched_detections)
    {
        this->_initiate_track(detections[detection_idx]);
    }
    // 用于删除被标记为删除状态的轨迹
    vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();)
    {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }

    // 用于更新跟踪器的度量模型。首先，通过遍历跟踪器中的轨迹对象，筛选出已确认状态的轨迹。然后，
    // 将这些轨迹的ID和特征向量存储到active_targets和tid_features中。
    // 接着，将轨迹的特征向量重置为空。最后，调用度量模型的partial_fit函数，传递轨迹的ID和特征向量，以更新度量模型

    // vector<int> active_targets;
    // vector<TRACKER_DATA> tid_features;
    // for (Track &track : tracks)
    // {
    //     if (track.is_confirmed() == false)
    //         continue;
    //     active_targets.push_back(track.track_id);
    //     tid_features.push_back(std::make_pair(track.track_id, track.features));
    //     FEATURESS t = FEATURESS(0, k_feature_dim);
    //     track.features = t;
    // }
    // this->metric->partial_fit(tid_features, active_targets);
}

void tracker::_match(const DETECTIONS &detections, TRACHER_MATCHD &res)
{
    // 将跟踪器中的轨迹分为已确认状态和未确认状态
    // vector<int> confirmed_tracks;
    // vector<int> unconfirmed_tracks;
    // int idx = 0;
    // for (Track &t : tracks)
    // {
    //     if (t.is_confirmed())
    //         confirmed_tracks.push_back(idx);
    //     else
    //         unconfirmed_tracks.push_back(idx);
    //     idx++;
    // }
    // 进行级联匹配
    // TRACHER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
    //     this, &tracker::gated_matric,
    //     this->metric->mating_threshold,
    //     this->max_age,
    //     this->tracks,
    //     detections,
    //     confirmed_tracks);
    
    //处理未匹配的轨迹。首先，将未确认轨迹的索引赋值给iou_track_candidates。然后，通过遍历matcha.unmatched_tracks中的未匹配轨迹索引，
    //如果轨迹的time_since_update为1（即上一次更新后的时间为1），则将该轨迹索引添加到iou_track_candidates中，
    //并从matcha.unmatched_tracks中删除该索引
    vector<int> iou_track_candidates;
    for (size_t i = 0; i < tracks.size(); i++) {
        iou_track_candidates.push_back(i);
    }
    // vector<int>::iterator it;
    // for (it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();)
    // {
    //     int idx = *it;
    //     if (tracks[idx].time_since_update == 1)
    //     { // push into unconfirmed
    //         iou_track_candidates.push_back(idx);
    //         it = matcha.unmatched_tracks.erase(it);
    //         continue;
    //     }
    //     ++it;
    // }

    // 进行最小成本匹配(匈牙利匹配)
    std::vector<int> detection_indices;
    for (size_t i = 0; i < detections.size(); i++) {
        detection_indices.push_back(i);
    }
    TRACHER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
        this, &tracker::iou_cost,
        this->max_iou_distance,
        this->tracks,
        detections,
        iou_track_candidates,
        detection_indices);

    //将匹配结果matcha和matchb中的匹配轨迹、未匹配轨迹和未匹配检测结果，分别添加到结果对象res的相应字段中
    // res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    // unmatched_tracks;
    // res.unmatched_tracks.assign(
    //     matcha.unmatched_tracks.begin(),
    //     matcha.unmatched_tracks.end());
    res.unmatched_tracks.insert(
        res.unmatched_tracks.end(),
        matchb.unmatched_tracks.begin(),
        matchb.unmatched_tracks.end());
    res.unmatched_detections.assign(
        matchb.unmatched_detections.begin(),
        matchb.unmatched_detections.end());
}

void tracker::_initiate_track(const DETECTION_ROW &detection)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
                                 this->max_age, detection.feature));
    _next_idx += 1;
}

DYNAMICM tracker::gated_matric(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    //创建了一个名为features的矩阵，用于存储检测结果的特征向量
    FEATURESS features(detection_indices.size(), k_feature_dim);
    int pos = 0;
    for (int i : detection_indices)
    {
        features.row(pos++) = dets[i].feature;
    }
    // targets的向量，用于存储跟踪器中的轨迹的ID
    vector<int> targets;
    for (int i : track_indices)
    {
        targets.push_back(tracks[i].track_id);
    }
    //调用了度量模型的distance函数，计算特征向量矩阵features与目标轨迹ID向量targets之间的距离矩阵
    DYNAMICM cost_matrix = this->metric->distance(features, targets);
    DYNAMICM res = linear_assignment::getInstance()->gate_cost_matrix(
        this->kf, cost_matrix, tracks, dets, track_indices,
        detection_indices);
    return res;
}

DYNAMICM
tracker::iou_cost(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    // 计算了成本矩阵的行数和列数，并创建了一个大小为rows × cols 的零矩阵cost_matrix，用于存储成本值
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    // 遍历轨迹索引track_indices，计算每个轨迹与检测结果之间的IoU成本
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows; i++)
    {
        int track_idx = track_indices[i];
        if (tracks[track_idx].time_since_update > this->max_age) // FIXME 最大寿命没有起效
        // if (tracks[track_idx].time_since_update > 1)
        {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }
        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        DETECTBOXSS candidates(csize, 4);
        for (int k = 0; k < csize; k++)
            candidates.row(k) = dets[detection_indices[k]].tlwh;
        Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

Eigen::VectorXf
tracker::iou(DETECTBOX &bbox, DETECTBOXSS &candidates) // 用于计算一个边界框（bbox）与一组候选边界框（candidates）之间的IoU
{
    // 将边界框的左上角坐标（bbox_tl_1、bbox_tl_2）和右下角坐标（bbox_br_1、bbox_br_2）提取出来，并计算边界框的面积（area_bbox）
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];
    // 将候选边界框的左上角坐标（candidates_tl）和右下角坐标（candidates_br）提取出来。
    //candidates是一个矩阵，每行表示一个候选边界框，前两列是左上角坐标，后两列是宽度和高度。
    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2);
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    // 这段代码通过遍历候选边界框，计算每个候选边界框与边界框之间的IoU值
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++)
    {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1;
        w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2;
        h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
    }
    //#ifdef MY_inner_DEBUG
    //        std::cout << res << std::endl;
    //#endif
    return res;
}
