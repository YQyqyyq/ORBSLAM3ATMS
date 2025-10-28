#include "TagManager.h"
#include "KeyFrame.h"
#include <apriltag.h>
#include <tag36h11.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include "Converter.h"

using namespace ORB_SLAM3;

// 静态成员初始化
apriltag_family_t* TagStorage::mpTagFamily = nullptr;
apriltag_detector_t* TagStorage::mpDetector = nullptr;

TagStorage& TagStorage::Instance() {
    static TagStorage instance;
    return instance;
}

TagStorage::TagStorage() 
    : mLoaded(false){
    // 构造时默认不初始化 AprilTag 检测器
}

TagStorage::~TagStorage() {
    DestroyDetector();
    DestroyDetectorPool();
}

void TagStorage::InitDetectorAndLoad(const std::string& filename) {
    if (mpDetector) return; // 已初始化
    // 创建 tag36h11 family
    mpTagFamily = tag36h11_create();
    // 创建检测器
    mpDetector = apriltag_detector_create();
    apriltag_detector_add_family_bits(mpDetector, mpTagFamily, 1);
    // 参数配置
    mpDetector->quad_decimate = 2.0;
    mpDetector->quad_sigma    = 0.0;
    mpDetector->nthreads      = 4;
    mpDetector->debug         = 0;
    mpDetector->refine_edges  = 1;

    int detector_pool_size = 4; 
        for (int i = 0; i < detector_pool_size; i++) {
        apriltag_detector_t* detector = apriltag_detector_create();
        apriltag_detector_add_family_bits(detector, mpTagFamily, 1);
        detector->quad_decimate = 2.0;
        detector->quad_sigma = 0.0;
        detector->nthreads = 1;
        detector->debug = 0;
        detector->refine_edges = 1;
        mDetectorPool.push_back(detector);
    }

    // 从txt文件载入已有Tag
    mLoaded = tagLoad(filename);
    if (mLoaded) {
        std::cout << "[TagStorage] Loaded tag poses from \"" << filename << "\".\n";
    } else {
        std::cout << "[TagStorage] No existing pose file found, will compute from observations.\n";
    }
}

apriltag_detector_t* TagStorage::GetDetector() {
    if (!mpDetector) InitDetectorAndLoad();
    return mpDetector;
}

void TagStorage::DestroyDetector() {
    if (mpDetector) {
        apriltag_detector_destroy(mpDetector);
        mpDetector = nullptr;
    }
    if (mpTagFamily) {
        tag36h11_destroy(mpTagFamily);
        mpTagFamily = nullptr;
    }
}

void TagStorage::InitDetectorPool(int pool_size) {
    if (!mpTagFamily) {
        InitDetectorAndLoad();
    }
    for (int i = 0; i < pool_size; i++) {
        apriltag_detector_t* detector = apriltag_detector_create();
        apriltag_detector_add_family_bits(detector, mpTagFamily, 1);
        detector->quad_decimate = 2.0;
        detector->quad_sigma = 0.0;
        detector->nthreads = 1;
        detector->debug = 0;
        detector->refine_edges = 1;
        mDetectorPool.push_back(detector);
    }
}

apriltag_detector_t* TagStorage::GetThreadDetector(int thread_id) {
    if (mDetectorPool.empty()) {
        InitDetectorPool(4); // 默认创建4个
    }
    return mDetectorPool[thread_id % mDetectorPool.size()];
}

void TagStorage::DestroyDetectorPool() {
    for (auto detector : mDetectorPool) {
        apriltag_detector_destroy(detector);
    }
    mDetectorPool.clear();
}


// ------- 具体操作函数 -------
void TagStorage::tagWrite(int id,
                           KeyFrame* pKF,
                           const Eigen::Matrix3d R_cam_tag,
                           const Eigen::Vector3d t_cam_tag,
                           int camID)
{
    if (!pKF) return; // 自检：空指针直接丢弃
    std::lock_guard<std::mutex> lock(mMutex);
    TagObs tempobs;
    tempobs.pKF = pKF;
    tempobs.R_cam_tag = R_cam_tag;
    tempobs.t_cam_tag = t_cam_tag;
    tempobs.camID = camID;
    mStorage[id].push_back(tempobs);
}

bool TagStorage::tagRead(int id,
                         Eigen::Matrix3d& R_w_tag_avg,
                         Eigen::Vector3d& t_w_tag_avg)
{
    // tagCleanup();//zyq 性能考虑
    std::lock_guard<std::mutex> lock(mMutex);

    if(mLoaded){
        auto itRt = mStorageRt.find(id);
        if (itRt != mStorageRt.end()) {
            R_w_tag_avg = itRt->second.first;
            t_w_tag_avg = itRt->second.second;
            return true;
        }
        else{
            return false;
        }
    }

    // 从观测累积计算
    auto itObs = mStorage.find(id);
    if (itObs == mStorage.end() || itObs->second.empty())
        return false;

    std::vector<Eigen::Quaterniond> qs;
    std::vector<Eigen::Vector3d> ts;
    std::vector<size_t> valid_indices; // 记录有效观测的索引

    for (size_t i = 0; i < itObs->second.size(); i++) {

        if (i >= 15)
            break;
        
        auto& obs = itObs->second[i];
        ORB_SLAM3::KeyFrame* pKF = obs.pKF;
        if (!pKF || pKF->isBad()) continue;

        Sophus::SE3f Twc_SE3 = pKF->GetPoseInverse(obs.camID);
        Eigen::Matrix4f Twc_mat = Twc_SE3.matrix();
        Eigen::Matrix3d R_w_c = Twc_mat.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d t_w_c = Twc_mat.block<3,1>(0,3).cast<double>();

        Eigen::Matrix3d R_w_tag = R_w_c * obs.R_cam_tag;
        Eigen::Vector3d t_w_tag = R_w_c * obs.t_cam_tag + t_w_c;

        qs.emplace_back(R_w_tag);
        ts.emplace_back(t_w_tag);
        valid_indices.push_back(i); // 记录这个观测在原始存储中的索引
    }

    if (qs.empty()) return false;

    Eigen::Quaterniond q_avg(0,0,0,0);
    for (auto& q : qs) q_avg.coeffs() += q.coeffs();
    q_avg.normalize();

    Eigen::Vector3d t_avg(0,0,0);
    for (auto& t : ts) t_avg += t;
    t_avg /= double(ts.size());

    std::vector<double> errs;
    errs.reserve(ts.size());
    for (auto& t : ts) {
        errs.push_back((t - t_avg).norm());
    }

    R_w_tag_avg = q_avg.toRotationMatrix();
    t_w_tag_avg = t_avg;

    mStorageRt[id] = std::make_pair(R_w_tag_avg, t_w_tag_avg);
    return true;
}

bool TagStorage::tagRead(int id,
                         Eigen::Matrix3d& R_w_tag_avg,
                         Eigen::Vector3d& t_w_tag_avg,
                         double& t_err_avg)
{
    std::lock_guard<std::mutex> lock(mMutex);

    // 观测累积计算
    auto itObs = mStorage.find(id);
    if (itObs == mStorage.end() || itObs->second.empty())
        return false;

    std::vector<Eigen::Quaterniond> qs;
    std::vector<Eigen::Vector3d> ts;

    for (auto& obs : itObs->second) {
        ORB_SLAM3::KeyFrame* pKF = obs.pKF;
        if (!pKF || pKF->isBad()) continue;

        Sophus::SE3f Twc_SE3 = pKF->GetPoseInverse(obs.camID);
        Eigen::Matrix4f Twc_mat = Twc_SE3.matrix();
        Eigen::Matrix3d R_w_c = Twc_mat.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d t_w_c = Twc_mat.block<3,1>(0,3).cast<double>();

        Eigen::Matrix3d R_w_tag = R_w_c * obs.R_cam_tag;
        Eigen::Vector3d t_w_tag = R_w_c * obs.t_cam_tag + t_w_c;

        qs.emplace_back(R_w_tag);
        ts.emplace_back(t_w_tag);
    }

    if (qs.empty()) return false;

    Eigen::Quaterniond q_avg(0,0,0,0);
    for (auto& q : qs) q_avg.coeffs() += q.coeffs();
    q_avg.normalize();

    Eigen::Vector3d t_avg(0,0,0);
    for (auto& t : ts) t_avg += t;
    t_avg /= double(ts.size());

    R_w_tag_avg = q_avg.toRotationMatrix();
    t_w_tag_avg = t_avg;

    std::vector<double> errs;
    errs.reserve(ts.size());
    for (auto& t : ts) {
        errs.push_back((t - t_avg).norm());
    }
    // 求最大和平均
    double t_err_max = *std::max_element(errs.begin(), errs.end());
    t_err_avg = std::accumulate(errs.begin(), errs.end(), 0.0) / errs.size();
    
    cout<< "id：" << id << "最大误差：" << t_err_max << "m 平均误差：" << t_err_avg << "m\n";
    return true;
}

bool TagStorage::tagRead4LC(int id,
                         Eigen::Matrix3d& R_w_tag_avg,
                         Eigen::Vector3d& t_w_tag_avg,
                         double& t_err_avg)
{
    std::lock_guard<std::mutex> lock(mMutex);

    auto itObs = mStorage.find(id);
    if (itObs == mStorage.end() || itObs->second.empty())
        return false;

    const auto& obsVec = itObs->second;
    size_t obsCount = obsVec.size();
    size_t numToExclude = 0;

    const auto& lastObs = obsVec[obsCount - 1];
    const auto& secondLastObs = obsVec[obsCount - 2];
    if (lastObs.pKF && secondLastObs.pKF && lastObs.pKF == secondLastObs.pKF) {
        numToExclude = 2;
    } else {
        numToExclude = 1;
    }

    std::vector<Eigen::Quaterniond> qs;
    std::vector<Eigen::Vector3d> ts;

    for (size_t i = 0; i < obsCount - numToExclude - 1; ++i) {

        if (i >= 15)
            break;
        
        const auto& obs = obsVec[i];
        ORB_SLAM3::KeyFrame* pKF = obs.pKF;
        if (!pKF || pKF->isBad()) continue;

        Sophus::SE3f Twc_SE3 = pKF->GetPoseInverse(obs.camID);
        Eigen::Matrix4f Twc_mat = Twc_SE3.matrix();
        Eigen::Matrix3d R_w_c = Twc_mat.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d t_w_c = Twc_mat.block<3,1>(0,3).cast<double>();

        Eigen::Matrix3d R_w_tag = R_w_c * obs.R_cam_tag;
        Eigen::Vector3d t_w_tag = R_w_c * obs.t_cam_tag + t_w_c;

        qs.emplace_back(R_w_tag);
        ts.emplace_back(t_w_tag);
    }

    if (qs.empty()) return false;

    Eigen::Quaterniond q_avg(0,0,0,0);
    for (const auto& q : qs) q_avg.coeffs() += q.coeffs();
    q_avg.normalize();

    Eigen::Vector3d t_avg(0,0,0);
    for (const auto& t : ts) t_avg += t;
    t_avg /= double(ts.size());

    std::vector<double> errs;
    errs.reserve(ts.size());
    for (const auto& t : ts) {
        errs.push_back((t - t_avg).norm());
    }
    
    t_err_avg = std::accumulate(errs.begin(), errs.end(), 0.0) / errs.size();
    
    R_w_tag_avg = q_avg.toRotationMatrix();
    t_w_tag_avg = t_avg;

    return true;
}



void TagStorage::tagCleanup() {
    std::lock_guard<std::mutex> lk(mMutex);
    for (auto it = mStorage.begin(); it != mStorage.end(); ) {
        auto& vec = it->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [](const TagObs& obs) {
            return (obs.pKF == nullptr) || (obs.pKF->isBad());
        }), vec.end());

        if (vec.empty()) {
            it = mStorage.erase(it);
        } else {
            ++it;
        }
    }
}

bool TagStorage::tagSave(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::trunc);
    if (!outFile) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }

    outFile << std::fixed << std::setprecision(15);

    for (const auto& entry : mStorageRt) {
        int id = entry.first;
        const auto& Rt = entry.second;
        const Eigen::Matrix3d& R = Rt.first;
        const Eigen::Vector3d& t = Rt.second;

        if (!isRtValid(R, t)) {
            std::cout << "跳过无效 ID: " << id << std::endl;
            continue;
        }

        outFile << id << " ";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                outFile << R(i, j) << " ";
            }
        }
        outFile << t(0) << " " << t(1) << " " << t(2) << "\n";
    }
    outFile.close();
    return true;
}

bool TagStorage::tagLoad(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    mStorageRt.clear();

    int id;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    while (inFile >> id) {
        // 读取 R 和 t
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (!(inFile >> R(i, j))) {
                    inFile.close();
                    return false;
                }
            }
        }

        if (!(inFile >> t(0) >> t(1) >> t(2))) {
            inFile.close();
            return false;
        }

        // 检查 Rt 是否有效，无效则跳过
        if (!isRtValid(R, t)) {
            continue;
        }

        // 存入 mStorageRt
        mStorageRt[id] = std::make_pair(R, t);
    }

    inFile.close();
    mLoaded = true;
    return true;
}

// 获取某个关键帧的所有Tag观测
std::map<int, std::vector<TagStorage::TagObs>> TagStorage::GetObservationsForKF(int kfId) {
    std::lock_guard<std::mutex> lock(mMutex);
    std::map<int, std::vector<TagObs>> result;
    for(auto& [tagId, obsVec] : mStorage) {
        for(auto& obs : obsVec) {
            if(obs.pKF && obs.pKF->mnId == kfId) {
                result[tagId].push_back(obs);
            }
        }
    }
    return result;
}

// 获取某个Tag的所有关键帧观测
std::vector<ORB_SLAM3::KeyFrame*> TagStorage::GetObservingKFForTag(int id)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::set<ORB_SLAM3::KeyFrame*> sKFs;

    auto it = mStorage.find(id);
    if (it != mStorage.end())
    {
        const auto& observations = it->second;
        for (const auto& obs : observations)
        {
            if (obs.pKF && !obs.pKF->isBad())
            {
                sKFs.insert(obs.pKF);
            }
        }
    }
    return std::vector<ORB_SLAM3::KeyFrame*>(sKFs.begin(), sKFs.end());
}

// 获取Tag的观测次数
int TagStorage::GetObservationCount(int tagId) {
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mStorage.find(tagId);
    return (it != mStorage.end()) ? it->second.size() : 0;
}

bool TagStorage::isRtValid(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, double eps) {
    // 检查 R 是否正交
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (!(R * R.transpose()).isApprox(I, eps)) {
        return false;
    }
    // 检查 t 是否为零向量
    if (t.norm() < eps) {
        return false;
    }
    return true;
}