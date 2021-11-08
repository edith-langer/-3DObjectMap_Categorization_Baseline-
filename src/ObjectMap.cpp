//#include "rtabmap/core/RtabmapExp.h" // DLL export/import defines
//#include "rtabmap/core/Memory.h" // DLL export/import defines

#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <opencv2/core/core.hpp>
#include "Transform.h"
#include "CameraModel.h"

#include <chrono>
#include <opencv2/core/types_c.h>
#include <regex>
#include <boost/filesystem.hpp>
#include <tuple>

#include "json.hpp"
using json = nlohmann::json;

using namespace std;
using namespace cv;

template<class T>
inline bool uIsFinite(const T& value)
{
#if _MSC_VER
    return _finite(value) != 0;
#else
    return std::isfinite(value);
#endif
}



struct PointXYZ { float x; float y; float z; };

float getDepth(
        const cv::Mat& depthImage,
        float x, float y,
        bool smoothing,
        float depthErrorRatio,
        bool estWithNeighborsIfNull = false)
{
    assert(!depthImage.empty());
    assert(depthImage.type() == CV_16UC1 || depthImage.type() == CV_32FC1);

    int u = int(x + 0.5f);
    int v = int(y + 0.5f);
    if (u == depthImage.cols && x<float(depthImage.cols))
    {
        u = depthImage.cols - 1;
    }
    if (v == depthImage.rows && y<float(depthImage.rows))
    {
        v = depthImage.rows - 1;
    }

    if (!(u >= 0 && u < depthImage.cols && v >= 0 && v < depthImage.rows))
    {
        printf("!(x >=0 && x<depthImage.cols && y >=0 && y<depthImage.rows) cond failed! returning bad point. (x=%f (u=%d), y=%f (v=%d), cols=%d, rows=%d)\n",
               x, u, y, v, depthImage.cols, depthImage.rows);
        return 0;
    }

    bool isInMM = depthImage.type() == CV_16UC1; // is in mm?

    // Inspired from RGBDFrame::getGaussianMixtureDistribution() method from
    // https://github.com/ccny-ros-pkg/rgbdtools/blob/master/src/rgbd_frame.cpp
    // Window weights:
    //  | 1 | 2 | 1 |
    //  | 2 | 4 | 2 |
    //  | 1 | 2 | 1 |
    int u_start = std::max(u - 1, 0);
    int v_start = std::max(v - 1, 0);
    int u_end = std::min(u + 1, depthImage.cols - 1);
    int v_end = std::min(v + 1, depthImage.rows - 1);

    float depth = 0.0f;
    if (isInMM)
    {
        if (depthImage.at<unsigned short>(v, u) > 0 &&
                depthImage.at<unsigned short>(v, u) < std::numeric_limits<unsigned short>::max())
        {
            depth = float(depthImage.at<unsigned short>(v, u)) * 0.001f;
        }
    }
    else
    {
        depth = depthImage.at<float>(v, u);
    }

    if ((depth == 0.0f || !uIsFinite(depth)) && estWithNeighborsIfNull)
    {
        // all cells no2 must be under the zError to be accepted
        float tmp = 0.0f;
        int count = 0;
        for (int uu = u_start; uu <= u_end; ++uu)
        {
            for (int vv = v_start; vv <= v_end; ++vv)
            {
                if ((uu == u && vv != v) || (uu != u && vv == v))
                {
                    float d = 0.0f;
                    if (isInMM)
                    {
                        if (depthImage.at<unsigned short>(vv, uu) > 0 &&
                                depthImage.at<unsigned short>(vv, uu) < std::numeric_limits<unsigned short>::max())
                        {
                            d = float(depthImage.at<unsigned short>(vv, uu)) * 0.001f;
                        }
                    }
                    else
                    {
                        d = depthImage.at<float>(vv, uu);
                    }
                    if (d != 0.0f && uIsFinite(d))
                    {
                        if (tmp == 0.0f)
                        {
                            tmp = d;
                            ++count;
                        }
                        else
                        {
                            float depthError = depthErrorRatio * tmp;
                            if (fabs(d - tmp / float(count)) < depthError)

                            {
                                tmp += d;
                                ++count;
                            }
                        }
                    }
                }
            }
        }
        if (count > 1)
        {
            depth = tmp / float(count);
        }
    }

    if (depth != 0.0f && uIsFinite(depth))
    {
        if (smoothing)
        {
            float sumWeights = 0.0f;
            float sumDepths = 0.0f;
            for (int uu = u_start; uu <= u_end; ++uu)
            {
                for (int vv = v_start; vv <= v_end; ++vv)
                {
                    if (!(uu == u && vv == v))
                    {
                        float d = 0.0f;
                        if (isInMM)
                        {
                            if (depthImage.at<unsigned short>(vv, uu) > 0 &&
                                    depthImage.at<unsigned short>(vv, uu) < std::numeric_limits<unsigned short>::max())
                            {
                                d = float(depthImage.at<unsigned short>(vv, uu)) * 0.001f;
                            }
                        }
                        else
                        {
                            d = depthImage.at<float>(vv, uu);
                        }

                        float depthError = depthErrorRatio * depth;

                        // ignore if not valid or depth difference is too high
                        if (d != 0.0f && uIsFinite(d) && fabs(d - depth) < depthError)
                        {
                            if (uu == u || vv == v)
                            {
                                sumWeights += 2.0f;
                                d *= 2.0f;
                            }
                            else
                            {
                                sumWeights += 1.0f;
                            }
                            sumDepths += d;
                        }
                    }
                }
            }
            // set window weight to center point
            depth *= 4.0f;
            sumWeights += 4.0f;

            // mean
            depth = (depth + sumDepths) / sumWeights;
        }
    }
    else
    {
        depth = 0;
    }
    return depth;
}


PointXYZ projectDepthTo3D(
        const cv::Mat& depthImage,
        float x, float y,
        float cx, float cy,
        float fx, float fy,
        bool smoothing,
        float depthErrorRatio = 0.02f)
{
    assert(depthImage.type() == CV_16UC1 || depthImage.type() == CV_32FC1);

    PointXYZ pt;

    float depth = getDepth(depthImage, x, y, smoothing, depthErrorRatio);
    //std::cout << depth << " " << depthImage.at<float>(x, y) << " " << depthImage.at<float>(y, x) << std::endl;
    if (depth > 0.0f)
    {
        // Use correct principal point from calibration
        cx = cx > 0.0f ? cx : float(depthImage.cols / 2) - 0.5f; //cameraInfo.K.at(2)
        cy = cy > 0.0f ? cy : float(depthImage.rows / 2) - 0.5f; //cameraInfo.K.at(5)

        // Fill in XYZ
        pt.x = (x - cx) * depth / fx;
        pt.y = (y - cy) * depth / fy;
        pt.z = depth;
    }
    else
    {
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
    }
    return pt;
}

// object label and its position on the map and 
typedef struct labelpos {
    string label;
    Point3d pm;
    Point2d cbb;
    float area;
    string feature_vector_path;
} LabelPos;

// node id and the object position on the map
typedef struct idpos{
    int id;
    Point3d pm;
    Point2d cbb;
    float area;
} IdPos;

typedef struct candobj{
    Point3d pm;
    Point2d cbb;
    float area;
    set<int> ids;
    map<string, int> labelHist;
    string label;
    string obj;
    std::vector<std::string> feature_vector_paths;
} CandObj;

struct Intrinsics {
    float cx;
    float cy;
    float fx;
    float fy;
    int width;
    int height;
};

struct DetObject {
    std::string rgb_path;
    std::vector<std::string> depth_paths; //we can have several depth images associated with one rgb image
    //because the depth is cut off according to the tables we are interested in
    Transform tf;

};


void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string> classes);
std::map<int, DetObject> readInputData(std::vector<std::tuple<std::string, std::string>> association_freiburg_vec);
string getMaxVotedLabel(CandObj& m);


int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point begin_total = std::chrono::steady_clock::now();

    //TODO
    std::string scene_path = argv[1];
    float xi = 0;// = atof(argv[4]);
    float yi = 0; // = atof(argv[5]);
    float zi = 0; // = atof(argv[6]);
    float roll = 0; // = atof(argv[7]);
    float pitch = 0; // = atof(argv[8]);
    float yaw = 0; // = atof(argv[9]);
    //cout << xi << " " << yi << " " << zi << " " << roll << " " << pitch << " " << yaw << endl;

    // TODO
    //freiburg-file (fusedPoses_i100_biggerLambda) and associations.txt needed. Skip first line in both files and then the lines can be read one after the other
    //use rgb id as key (to find the pre-computed detections) and save the path to rgb and depth frame
    std::vector<std::tuple<std::string, std::string> > association_freiburg_vec; //stores for all planes the path to the freiburg and association file
    const std::regex ass_filter("associations_[0-9]+.txt");
    std::string planes_path = scene_path + "/planes/";
    for(auto & plane_folder_path : boost::filesystem::directory_iterator(planes_path))
    {
        if (boost::filesystem::is_directory(plane_folder_path.status()))
        {
            for(auto & plane_files : boost::filesystem::directory_iterator(plane_folder_path)) {
                if(std::regex_match(plane_files.path().filename().string(), ass_filter ) ) {
                    std::string ass_name = plane_files.path().stem().string();
                    std::string ass_nr = ass_name.substr(ass_name.find_last_of("_")+1);
                    std::string freiburg_path = plane_folder_path.path().string() + "/table_" + ass_nr + "_fusedPoses_i100_biggerLambda.freiburg";
                    if (!boost::filesystem::exists(freiburg_path)) {
                        std::cerr << "Couldn't find file " << freiburg_path << std::endl;
                    }
                    std::tuple<std::string, std::string> ass_freiburg = std::make_tuple(plane_files.path().string(), freiburg_path);
                    association_freiburg_vec.push_back(ass_freiburg);
                }
            }
        }
    }

    std::map<int, DetObject> input_data = readInputData(association_freiburg_vec);

    //intrinsics from Sasha's Asus camera
    Intrinsics intrinsics = Intrinsics();
    intrinsics.fx = 538.391;
    intrinsics.fy = 538.085;
    intrinsics.cx = 315.307;
    intrinsics.cy = 233.048;
    intrinsics.height = 480;
    intrinsics.width = 640;


    map<int, vector<LabelPos>> idLabelMap;
    map<string, vector<IdPos>> labelIdMap;
    map<int, CandObj> candObjMap;


    vector<string> classes;

    string classesFile = "/home/edith/Projects/SpatioTemporal3DObject_Baseline/valid_classes.txt";

    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)){
        if (line.rfind("#", 0) != 0)
            classes.push_back(line);
    }
    ifs.close();

    // transform from camera to base
    Transform tbc(0.00220318, -0.00605847, 0.999979, 0.216234, -0.999998, -0.000013348,
                  0.00220314, 0.0222121, 0.0, -0.999982, -0.00605849, 1.20972);
    tbc.setIdentity();



    std::map<int, DetObject>::iterator input_it = input_data.begin();

    std::chrono::steady_clock::time_point begin_obj_det = std::chrono::steady_clock::now();
    while ((input_it != input_data.end()))
    {
        std::vector<cv::Mat> depth_imgs;
        for (std::string depth_path : input_it->second.depth_paths) {
            cv::Mat depth_img = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
            depth_imgs.push_back(depth_img);
        }

        // read a JSON file
        std::string detection_folder_path = (boost::filesystem::path(input_it->second.rgb_path).parent_path() /
                                             boost::filesystem::path(input_it->second.rgb_path).stem()).string() + "_detections/";
        std::ifstream detection_path(detection_folder_path + boost::filesystem::path(input_it->second.rgb_path).stem().string() + ".json");
        json json_file;
        detection_path >> json_file;

        auto detections = json_file["detections"];
        // iterate the array
        for (json::iterator json_it = detections.begin(); json_it != detections.end(); ++json_it) {
            auto detection = *json_it;
            int id = detection["id"].get<int>();
            std::string label = detection["label"].get<std::string>();
            float score = std::stof(detection["score"].get<std::string>());
            auto bb = detection["bb"];
            int minX = bb["minY"].get<int>();
            int minY = bb["minX"].get<int>();
            int maxX = bb["maxY"].get<int>();
            int maxY = bb["maxX"].get<int>();

            //if detection is not in the list of valid classes, we reject it
            if (std::find(classes.begin(), classes.end(), label) == classes.end())
                continue;

            //if score is less than conf, we reject the detection
            if (score < 0.9)
                continue;


            Point2d bb_center(minX + (maxX-minX)/2, minY + (maxY-minY)/2);

            //iterate through all depth images and check if depth point is valid
            //if the point is invalid in all depth images, we are not interested in the detection because it is not close to the planes
            PointXYZ p_3d_cam;
            p_3d_cam.x = p_3d_cam.y = p_3d_cam.z = std::numeric_limits<float>::quiet_NaN();
            for (cv::Mat depth_img : depth_imgs) {
                PointXYZ p_ = projectDepthTo3D(depth_img, bb_center.x, bb_center.y,
                                               intrinsics.cx, intrinsics.cy,
                                               intrinsics.fx, intrinsics.fy,
                                               false);
                if (p_.x != std::numeric_limits<float>::quiet_NaN() || p_.y != std::numeric_limits<float>::quiet_NaN() && p_.z != std::numeric_limits<float>::quiet_NaN()) {
                    p_3d_cam = p_;
                    break;
                }
            }

            //check for nans; the detection is not in an area we are interested in
            if (std::isnan(p_3d_cam.x) || std::isnan(p_3d_cam.y) || std::isnan(p_3d_cam.z))
                continue;


            // converting p to rtabmap::Transform. it's in camera's frame
            Transform p(p_3d_cam.x, p_3d_cam.y, p_3d_cam.z, 0, 0, 0);

            // map coordinates of p: tmb * tbc * p.
            // tmb is the transform from base to map, which is just the robot's pose.
            //el: the poses in the freiburg-file are the camera poses in map-frame
            Transform pm(input_it->second.tf * tbc * p);

            float bb_area = (maxX-minX) *  (maxY-minY);

            string fv_path = detection_folder_path + std::to_string(id) + ".feature";

            if (idLabelMap[input_it->first].empty())
                idLabelMap[input_it->first] = vector<LabelPos>();
            LabelPos x = {label, Point3d(pm.x(), pm.y(), pm.z()), bb_center, bb_area, fv_path};
            idLabelMap[input_it->first].push_back(x);

            if (labelIdMap[label].empty())
                labelIdMap[label] = vector<IdPos>();
            IdPos z = {input_it->first, Point3d(pm.x(), pm.y(), pm.z()), bb_center, bb_area};
            labelIdMap[label].push_back(z);
        }

        input_it++;
    }
    std::chrono::steady_clock::time_point end_obj_det = std::chrono::steady_clock::now();
    //std::cout << "Time difference obj_det = " <<
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_obj_det - begin_obj_det).count()
    //    << "[ms]" << std::endl;



    std::chrono::steady_clock::time_point begin_temp = std::chrono::steady_clock::now();
    int lastId = -1;
    int lastId2 = -1;
    int lastId3 = -1;

    // area threshold. if there is a change of less than aTh in the bb area,
    // in different frames, it's probably the same object
    float aTh = 0.2; //0.3

    // real world distance threshold in meters
    float dTh = 0.3;//0.8; // 1.0
    //float dTh = 7; // 1.0


    // bb center threshold
    //el: value used in paper is 0.0005 --> 154 px
    double bbcTh = 0.0001 * (640 * 480); // 30 px

    //int cnt = 0;
    //while (it2 != idLabelMap.end())
    input_it = input_data.begin();
    while (input_it != input_data.end())
    {
        int currId = input_it->first;
        //cout << "id " << currId << endl;
        //cout << "labels pos " << endl;
        //vector<LabelPos> v = it2->second;
        vector<LabelPos> v = idLabelMap[currId];
        // if first frame, all detected objects are candidates.
        if (candObjMap.empty())
        {
            for (int i = 0; i < v.size(); i++)
            {
                LabelPos currLabelPos = v[i];

                candObjMap[i].labelHist[currLabelPos.label]++;
                candObjMap[i].ids.insert(currId);
                candObjMap[i].pm = currLabelPos.pm;
                //candObjMap[i].pm.push_back(currLabelPos.pm);
                candObjMap[i].cbb = currLabelPos.cbb;
                candObjMap[i].area = currLabelPos.area;
                candObjMap[i].feature_vector_paths.push_back(currLabelPos.feature_vector_path);
                //cout << v[i].label << " " << v[i].cbb.x << " " << v[i].cbb.y
                //<< " " << v[i].area << endl;
            }
        }
        // from the second frame on, now we need to check objects association
        else
            //else if (it2 != idLabelMap.begin())
        {
            set<int> prohibitedCands;
            if (currId == 3540)
                cout << endl;
            for (int i = 0; i < v.size(); i++)
            {
                LabelPos currLabelPos = v[i];
                int j = 0;
                bool hasSameLabel = false;
                int index = -1;
                for (j = 0; j < candObjMap.size(); j++)
                {
                    set<int> currSet = candObjMap[j].ids;
                    //if ((currSet.find(lastId) != currSet.end()) || (currSet.find(lastId2) != currSet.end()))
                    if (((currSet.find(lastId) != currSet.end()) || (currSet.find(lastId2) != currSet.end())
                         || (currSet.find(lastId3) != currSet.end()))
                            && (prohibitedCands.find(j) == prohibitedCands.end()))
                    {
                        float dist = sqrt(powf(candObjMap[j].cbb.x - currLabelPos.cbb.x, 2) +
                                          powf(candObjMap[j].cbb.y - currLabelPos.cbb.y, 2));
                        if (dist < bbcTh && (fabs(candObjMap[j].area - currLabelPos.area)/currLabelPos.area < aTh))
                        {
                            if (currLabelPos.label == getMaxVotedLabel(candObjMap[j]))
                            {
                                hasSameLabel = true;
                                candObjMap[j].ids.insert(currId);
                                candObjMap[j].area = (candObjMap[j].area + currLabelPos.area) / 2;
                                candObjMap[j].labelHist[currLabelPos.label]++;
                                //candObjMap[j].pm = (candObjMap[j].pm + currLabelPos.pm) / 2;
                                candObjMap[j].pm.x = candObjMap[j].pm.x +
                                        (currLabelPos.pm.x - candObjMap[j].pm.x)/candObjMap[j].ids.size();
                                candObjMap[j].pm.y = candObjMap[j].pm.y +
                                        (currLabelPos.pm.y - candObjMap[j].pm.y)/candObjMap[j].ids.size();
                                candObjMap[j].pm.z = candObjMap[j].pm.z +
                                        (currLabelPos.pm.z - candObjMap[j].pm.z)/candObjMap[j].ids.size();
                                candObjMap[j].cbb = currLabelPos.cbb;
                                candObjMap[j].feature_vector_paths.push_back(currLabelPos.feature_vector_path);
                                prohibitedCands.insert(j);
                                break;
                            }
                            else
                                index = j;
                        }
                    }
                }
                // if j loop reached the end without breaking, it means that
                // there was no association. thus, candObjMap grows
                if (j == candObjMap.size())
                {
                    if (index == -1)
                    {
                        candObjMap[j].ids.insert(currId);
                        candObjMap[j].area = currLabelPos.area;
                        candObjMap[j].labelHist[currLabelPos.label]++;
                        candObjMap[j].pm = currLabelPos.pm;
                        candObjMap[j].cbb = currLabelPos.cbb;
                        candObjMap[j].feature_vector_paths.push_back(currLabelPos.feature_vector_path);
                    }
                    else
                    {
                        candObjMap[index].ids.insert(currId);
                        candObjMap[index].area = (candObjMap[index].area + currLabelPos.area) / 2;
                        candObjMap[index].labelHist[currLabelPos.label]++;
                        //candObjMap[index].pm = (candObjMap[index].pm + currLabelPos.pm) / 2;
                        candObjMap[index].pm.x = candObjMap[index].pm.x +
                                (currLabelPos.pm.x - candObjMap[index].pm.x)/candObjMap[index].ids.size();
                        candObjMap[index].pm.y = candObjMap[index].pm.y +
                                (currLabelPos.pm.y - candObjMap[index].pm.y)/candObjMap[index].ids.size();
                        candObjMap[index].pm.z = candObjMap[index].pm.z +
                                (currLabelPos.pm.z - candObjMap[index].pm.z)/candObjMap[index].ids.size();
                        candObjMap[index].cbb = currLabelPos.cbb;
                        candObjMap[index].feature_vector_paths.push_back(currLabelPos.feature_vector_path);
                        prohibitedCands.insert(index);
                    }
                }
            }
        }
        lastId3 = lastId2;
        lastId2 = lastId;
        lastId = currId;
        //it2++;
        input_it++;
    }



    std::chrono::steady_clock::time_point end_temp = std::chrono::steady_clock::now();
    //std::cout << "Time difference temporal association = " <<
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_temp - begin_temp).count()
    //    << "[ms]" << std::endl;
    //std::cout << "Time difference temporal association = " <<
    //    std::chrono::duration_cast<std::chrono::microseconds>(end_temp - begin_temp).count()
    //    << "[us]" << std::endl;

    //cout << "000" << endl;

    //cout << "CANDOBJMAP size " << candObjMap.size() << endl;


    // now to the spatial association
    //
    std::chrono::steady_clock::time_point begin_spat = std::chrono::steady_clock::now();

    map<string, CandObj> newCandObjMap;
    candObjMap[0].label = getMaxVotedLabel(candObjMap[0]);

    map<string, int> labelCount;
    bool match = false;
    int minQty = 3;

    for (auto it = candObjMap.begin(); it != candObjMap.end();)
    {
        //cout << candObjMap.size() << endl;
        CandObj currCand = it->second;
        if (currCand.ids.size() < minQty)
        {
            it = candObjMap.erase(it);
            continue;
        }
        currCand.label = getMaxVotedLabel(currCand);
        //cout << currCand.label << endl;
        if (labelCount.find(currCand.label) == labelCount.end())
            labelCount[currCand.label] = 0;
        //for (int k = j+1; k < candObjMap.size(); k++)
        for (auto itt = next(it, 1); itt != candObjMap.end();)
        {
            bool erased = false;
            //CandObj testCand = candObjMap[k];
            CandObj testCand = itt->second;
            if (testCand.label.empty())
                testCand.label = getMaxVotedLabel(testCand);
            if (testCand.ids.size() < minQty)
            {
                itt = candObjMap.erase(itt);
                continue;
            }
            set<int> intersect;
            set_intersection(testCand.ids.begin(), testCand.ids.end(), currCand.ids.begin(),
                             currCand.ids.end(), inserter(intersect, intersect.begin()));
            //if (currCand.label == testCand.label)
            if ((currCand.label == testCand.label) && (intersect.empty()))

            {
                float dist = sqrt(powf(currCand.pm.x - testCand.pm.x, 2) +
                                  powf(currCand.pm.y - testCand.pm.y, 2) +
                                  powf(currCand.pm.z - testCand.pm.z, 2));
                //if ((dist <= dTh) || (dist <= 0.1*sqrt(currCand.area)))
                if (dist <= dTh)
                    //if (dist <= (m*currCand.area/(480*640) + b))
                {
                    CandObj tmp;
                    tmp.ids.insert(currCand.ids.begin(), currCand.ids.end());
                    tmp.ids.insert(testCand.ids.begin(), testCand.ids.end());
                    tmp.label = currCand.label;
                    //tmp.pm = (currCand.pm + testCand.pm) / 2;

                    float cWeight = (float)(currCand.ids.size()) / ((float)currCand.ids.size() + testCand.ids.size());
                    float tWeight = (float)(testCand.ids.size()) / ((float)currCand.ids.size() + testCand.ids.size());
                    tmp.pm = (cWeight * currCand.pm + tWeight * testCand.pm)/(cWeight + tWeight);
                    tmp.area = (currCand.area + testCand.area) / 2;

                    tmp.feature_vector_paths.insert(tmp.feature_vector_paths.end(), currCand.feature_vector_paths.begin(), currCand.feature_vector_paths.end());
                    tmp.feature_vector_paths.insert(tmp.feature_vector_paths.end(), testCand.feature_vector_paths.begin(), testCand.feature_vector_paths.end());

                    string name = currCand.label + to_string(labelCount[currCand.label]);
                    string prevName = currCand.label + to_string(labelCount[currCand.label]-1);

                    //cout << "currcandobj " << currCand.obj << endl;
                    //cout << "testcandobj " << testCand.obj << endl;
                    if (newCandObjMap.find(prevName) == newCandObjMap.end())
                    {
                        //cout << prevName << endl;
                        newCandObjMap[name] = tmp;
                        labelCount[currCand.label]++;
                        currCand.obj = name;
                        tmp.obj = name;
                    }
                    else
                    {
                        map<string, CandObj>::iterator itn = newCandObjMap.begin();
                        while(itn != newCandObjMap.end())
                        {
                            CandObj nco = itn->second;
                            string inst = itn->first;
                            if (inst == currCand.obj)
                            {
                                //cout << "inst= " << inst << endl;
                                newCandObjMap[currCand.obj] = tmp;
                                tmp.obj = inst;
                                break;
                            }
                            itn++;
                        }

                        if (itn == newCandObjMap.end())
                        {
                            //cout << "fora while = " << name << endl;
                            tmp.obj = name;
                            newCandObjMap[name] = tmp;
                            labelCount[currCand.label]++;
                            tmp.obj = name;
                            currCand.obj = name;
                        }
                    }

                    itt = candObjMap.erase(itt);
                    currCand = tmp;
                    //k--;
                    match = true;
                    erased = true;
                }
            }
            if (!erased)
            {
                //cout << "itt++" << endl;
                itt++;
            }
        }
        // if no match was found, the object still exists. just copy.
        // (if it's been seen in at least 2 different views)
        if (!match)
        {
            if (currCand.ids.size() > 1)
            {
                newCandObjMap[currCand.label + to_string(labelCount[currCand.label])] = currCand;
                currCand.obj = currCand.label + to_string(labelCount[currCand.label]);
                //newCandObjMap[currCand.label + to_string(labelCount[currCand.label])]
                labelCount[currCand.label]++;
            }
        }
        match = false;
        //cout << "it++" << endl;
        it++;
    }


    //cout << "NEWCANDOBJMAP tamanho " << newCandObjMap.size() << endl;
    map<string, CandObj>::iterator itnc = newCandObjMap.begin();

    // final transform to align world axes from ROS and gazebo
    Transform t(xi, yi, zi, roll, pitch, yaw);

    std::string result_folder = scene_path + "/baseline_result/";
    boost::filesystem::create_directory(result_folder);

    std::ofstream obj_map_file;
    obj_map_file.open(result_folder + "/3d_obj_map.txt");

    while (itnc != newCandObjMap.end())
    {
        //cout << "label " << itnc->first << endl;
        //cout << itnc->first << " ";
        CandObj obj = itnc->second;
        set<int>::iterator its = obj.ids.begin();
        //cout << "ids ";
        while (its != obj.ids.end())
        {
            //cout << *its << " ";
            its++;
        }
        Transform q(obj.pm.x, obj.pm.y, obj.pm.z, 0, 0, 0);
        Transform r = t * q;
        obj_map_file << "[" << r.x() << "," << r.y() << "," << r.z() << "]" << ";" << itnc->first << "\n";

        //write all feature vector paths to a txt file for each detection to be able to perform matching
        std::ofstream obj_fv_file;
        obj_fv_file.open(result_folder + "/" + itnc->first +"_fv_paths.txt");
        for (std::string fv_path : obj.feature_vector_paths)
            obj_fv_file << fv_path << "\n";
        obj_fv_file.close();

        itnc++;
    }
    obj_map_file.close();



    std::chrono::steady_clock::time_point end_spat= std::chrono::steady_clock::now();
    //std::cout << "Time difference spatial association = " <<
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_spat- begin_spat).count()
    //    << "[ms]" << std::endl;
    //std::cout << "Time difference spatial association = " <<
    //    std::chrono::duration_cast<std::chrono::microseconds>(end_spat- begin_spat).count()
    //    << "[us]" << std::endl;

    /*
    map<string, vector<IdPos>>::iterator it3 = labelIdMap.begin();
    while (it3 != labelIdMap.end())
    {
        cout << "label " << it3->first << endl;
        cout << "id pos " << endl;
        vector<IdPos> v = it3->second;
        for (int i = 0; i < v.size(); i++)
        {

            //cout << v[i].id << " " << v[i].p << endl;
        }
        it3++;
    }
    */

    std::chrono::steady_clock::time_point end_total = std::chrono::steady_clock::now();
    //std::cout << "Time difference total = " <<
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_total - begin_total).count()
    //    << "[ms]" << std::endl;
    return 0;
}



string getMaxVotedLabel(CandObj& m)
{
    map<string, int>::iterator it = m.labelHist.begin();
    int currMax = it->second;
    string currMaxLabel = it->first;
    it++;
    while (it != m.labelHist.end())
    {
        if (it->second > currMax)
        {
            currMax = it->second;
            currMaxLabel = it->first;
        }
        it++;
    }
    return currMaxLabel;
}


std::map<int, DetObject> readInputData(std::vector<std::tuple<std::string, std::string>> association_freiburg_vec) {
    std::map<int, DetObject> input_data;
    for (std::tuple<std::string, std::string> t : association_freiburg_vec) {
        const string ass_path = std::get<0>(t);
        const string freiburg_path = std::get<1>(t);

        //freiburg format is "timestamp tx ty tz qx qy qz qw"
        //we skip the first line as it contains 0 0 0 0 0 0 1
        std::string timestamp;
        float x, y, z, qx, qy, qz, qw;
        std::ifstream freiburg_file(freiburg_path);

        //association format is "timestamp_depth path_depth timestamp_rgb path_rgb"
        std::string timestamp_depth, path_depth, timestamp_rgb, path_rgb;
        std::ifstream ass_file(ass_path);

        bool isFirstLine = true;
        while (freiburg_file >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw) {
            if (isFirstLine) {
                ass_file >> timestamp_depth >> path_depth >> timestamp_rgb >> path_rgb;
                isFirstLine = false;
                continue;
            }

            Transform tf(x, y, z, qx, qy, qz, qw);

            ass_file >> timestamp_depth >> path_depth >> timestamp_rgb >> path_rgb;
            if (timestamp != timestamp_depth) {
                std::cerr << "Something is weird while reading freiburg and association files. The timestamp should be the same, but it is " << timestamp << " and " << timestamp_depth << std::endl;
                exit(-1);
            }

            //extract the rgb image number e.g "../../rgb/rgb00715.png" --> 715
            int rgb_nr = std::stoi(path_rgb.substr(path_rgb.find_last_of("/")+4, 5));

            //create absolut paths for rgb and depth images
            boost::filesystem::path rgb_absolute_path = boost::filesystem::canonical(path_rgb,  boost::filesystem::path(ass_path).parent_path()); //canonical removes .. in paths
            boost::filesystem::path depth_absolute_path = boost::filesystem::canonical(path_depth,  boost::filesystem::path(ass_path).parent_path()); //canonical removes .. in paths

            input_data[rgb_nr].rgb_path = rgb_absolute_path.string();
            input_data[rgb_nr].depth_paths.push_back(depth_absolute_path.string());
            input_data[rgb_nr].tf = tf;
        }
    }
    return input_data;
}
