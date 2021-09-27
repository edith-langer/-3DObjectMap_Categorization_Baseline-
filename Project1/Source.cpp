//#include "rtabmap/core/RtabmapExp.h" // DLL export/import defines
//#include "rtabmap/core/Memory.h" // DLL export/import defines

#include <iostream>
#include <fstream>>
#include <string>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "Transform.h"
#include "CameraModel.h"
//#include "rtabmap/utilite/UMutex.h"
//#include "rtabmap/utilite/UThreadNode.h"
//#include "rtabmap/core/Parameters.h"
//#include "rtabmap/core/SensorData.h"
//#include <rtabmap/core/Statistics.h>
//#include "rtabmap/core/DBReader.h"
//#include "rtabmap/core/Parameters.h"
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl/pcl_base.h>
//#include <pcl/TextureMesh.h>
//#include "rtabmap/core/util3d.h"
//
//#include <rtabmap/core/Transform.h>
//#include <rtabmap/core/Link.h>
#include <chrono>
#include <opencv2/core/types_c.h>
using namespace rtabmap;
using namespace std;
using namespace cv;
using namespace dnn;

template<class T>
inline bool uIsFinite(const T& value)
{
#if _MSC_VER
    return _finite(value) != 0;
#else
    return std::isfinite(value);
#endif
}

struct SensorData
{
    Mat imageRaw() { // TODO }
        throw "bad";
        return Mat();
    }

    Mat depthRaw() { // TODO }
        throw "bad";
        return Mat();
    }
};

class Mem {
public:
    SensorData getSignatureDataConst(int id) {
        // TODO
        return SensorData();
    }
};


typedef struct PointXYZ { float x; float y; float z; };

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
        printf("!(x >=0 && x<depthImage.cols && y >=0 && y<depthImage.rows) cond failed! returning bad point. (x=%f (u=%d), y=%f (v=%d), cols=%d, rows=%d)",
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
} CandObj;

typedef struct Intrinsics {
    float cx;
    float cy;
    float fx;
    float fy;
    int width;
    int height;
};

//void postprocess(Mat& frame, const vector<Mat>& outs, double &confidence, Point2d &box, string target, vector<string> classes,
//        bool edgesValid);
void postprocess(Mat& frame, const vector<Mat>& outs, vector<float>& confidences, 
        vector<Point2d>& centers, const vector<string>& classes,
        bool edgesValid, vector<string>& detLabels, vector<float>& areas);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string> classes);
vector<String> getOutputsNames(const Net& net);
//string getMaxVotedLabel(CandObj& m);

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

int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point begin_total = std::chrono::steady_clock::now();

    std::string dbPath = argv[1];
    string posesFile = argv[2];
    // dir in which to save the processed images
    string outdir = argv[3];
    float xi = atof(argv[4]);
    float yi = atof(argv[5]);
    float zi = atof(argv[6]);
    float roll = atof(argv[7]);
    float pitch = atof(argv[8]);
    float yaw = atof(argv[9]);
    //cout << xi << " " << yi << " " << zi << " " << roll << " " << pitch << " " << yaw << endl; 

    // TODO
    Mem* mem = new Mem(); // fill in data
    Intrinsics intrinsics = Intrinsics();
    intrinsics.cx = 0;

    ifstream poses(posesFile);

    if (!poses.is_open())
    {
        cout << "File not open! Aborting...\n" << endl;
        return 1;
    }

    //Memory* mem = new Memory(param);
    //mem->init(dbPath, false, param, true);
    //std::set<int> ids = mem->getAllSignatureIds(true);
    //mem->init(dbPath, false, param, true);

    // TODO
    std::set<int> ids;

    Mat frame;
    Mat blob;
    SensorData data;

    map<int, vector<LabelPos>> idLabelMap;
    map<string, vector<IdPos>> labelIdMap;
    map<int, CandObj> candObjMap;

    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image
    vector<string> classes;
    int confidence_interval = 3;
    int frameIndex = 0;
    //double confidence;
    vector<float> confidences;
    vector<float> areas;
    //Point2d box;
    vector<Point2d> centers; 
    vector<string> detLabels;

    string classesFile = "/home/pal/felipe_ws/find_object/yolo/coco.names";
    
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)){ 
      classes.push_back(line);
    }

    
    // Give the configuration and weight files for the model
    String modelConfiguration = "/home/pal/felipe_ws/find_object/yolo/yolov3.cfg";
    String modelWeights = "/home/pal/felipe_ws/find_object/yolo/yolov3.weights";

    // Load the network
    //cout << "Loading neural model..." << endl;
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);


    // transform from camera to base
    Transform tbc(0.00220318, -0.00605847, 0.999979, 0.216234, -0.999998, -0.000013348,
            0.00220314, 0.0222121, 0.0, -0.999982, -0.00605849, 1.20972); 
    tbc.setIdentity();

    /*
camera to base_footprint transform
0.00220318   -0.00605847  0.999979     0.216234     
-0.999998    -1.3348e-05  0.00220314   0.0222121    
-2.95319e-14 -0.999982    -0.00605849  1.20972 

camera info
D: [1e-08, 1e-08, 1e-08, 1e-08, 1e-08]
K: [522.1910329546544, 0.0, 320.5, 0.0, 522.1910329546544, 240.5, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [522.1910329546544, 0.0, 320.5, -0.0, 0.0, 522.1910329546544, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]

fx 522.1910329546544
cx 320.5
fy 522.1910329546544
cy 240.5

frames 215 216

*/

    //TODO rgb and depht need same ide - associsations.txt

    std::set<int>::iterator it = ids.begin();
    //int framesBack = 3;
    //vector<int> multipliers;
    //vector<vector<Point2d>> objectsRaw;
    //vector<Point2d> objects;

    Transform pose;
    int mapid, weight;
    string l;
    double stamp;
    Transform gtruth;
    vector<float> vel;
 

    vector<CameraModel> cameraModels;

    //cout << "ids size " << ids.size() << endl;

    std::chrono::steady_clock::time_point begin_obj_det = std::chrono::steady_clock::now();
    int contador = 0;
    while ((it != ids.end()))
    //while ((it != ids.end()) && (contador < 200))
    {
        //cout << "testando\n\n" << endl;

        // TODO get stuff
        // mem->getNodeInfo(*it, pose, mapid, weight, l, stamp, gtruth, vel, gps, true);

        SensorData data = mem->getSignatureDataConst(*it);
        //data.uncompressData();


        // for kitti
        //cv::Mat K = (cv::Mat_<double>(3,3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);
        //cv::Mat D = (cv::Mat_<double>(1,5) << 1e-8, 1e-8, 1e-8, 1e-8, 1e-8);
        //cv::Mat R = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        //cv::Mat P = (cv::Mat_<double>(3,4) << 718.856, 0, 607.1928, 45.38225, 0, 718.856, 185.2157, -0.1130887, 0, 0, 
        //        1, 0.003779761);
        //CameraModel cm("nome", cv::Size(0,0), K, D, R, P);
        //cameraModels.push_back(cm);
        // end kitti
        //cout << cameraModels[0].fx() << endl;
        //cout << cameraModels[0].fy() << endl;
        //cout << cameraModels[0].cx() << endl;
        //cout << cameraModels[0].cy() << endl;

        frame = data.imageRaw();

        //cout << "empty " << frame.empty() << endl;

        float m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34;            
        poses >> m11 >> m12 >> m13 >> m14 >> m21 >> m22 >> m23 >> m24 >>
            m31 >> m32 >> m33 >> m34;
        Transform pose(m11, m12, m13, 
                m14, m21, m22, m23,
                m24, m31, m32, m33, m34);


        /*
        string line;
        if (posesFile.is_open())
        {
            getline(posesFile, line);
        }
        */

        //if (!frame.empty() && ((*it >= 272) && (*it <= 297)))
        //if (!frame.empty() && ((*it >= 127) && (*it <= 312)))
        //if (!frame.empty() && ((*it >= 22) && (*it <= 45)))
        //if (!frame.empty() && ((*it >= 426) && (*it <= 492)))
        if (!frame.empty() && !data.depthRaw().empty())
        //if (!frame.empty() && !data.laserScanRaw().data().empty())
        {
            contador++;

            //cout << "testando333\n\n" << endl;

            //cout << *it << endl;
            //cout << pose.prettyPrint() << endl;
            blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);

            //cout << "testando4\n\n" << endl;

            net.setInput(blob);
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));
            //postprocess(frame, outs, confidence, box, "sofa", classes, true);
            postprocess(frame, outs, confidences, centers, classes, false, detLabels, areas);
            //postprocess(frame, outs, confidences, centers, classes, true, detLabels, areas);

            //cout << "testando5\n\n" << endl;

            //cout << detLabels.size() << " " << areas.size() << " " << classes.size() << " " << 
            //    centers.size() << endl;

            std::stringstream sstm;
            sstm << outdir;
            if (*it < 10)
                sstm << "000";
            else if (*it < 100)
                sstm << "00";
            else if (*it < 1000)
                sstm << "0";
            sstm << *it << ".jpg";
            string fileName = sstm.str();  
            imwrite(fileName, frame);

            //cout << "testando6\n\n" << endl;
            
            bool saved = false;
            for (int i = 0; i < detLabels.size(); i++)
            {
                PointXYZ p_ = projectDepthTo3D(data.depthRaw(), centers[i].x, centers[i].y, 
                //pcl::PointXYZ p_ = rtabmap::util3d::projectDepthTo3D(data.laserScanRaw().data(), centers[i].x, centers[i].y, 
                        //320.5, 240.5,
                        //522.1910329546544, 522.1910329546544,
                        intrinsics.cx, intrinsics.cy,
                        intrinsics.fx, intrinsics.fy,
                        false);

                //cout << "testando7\n\n" << endl;

                if ((p_.x*p_.x + p_.y*p_.y + p_.z*p_.z) <= 100) //16
                {

                    // converting p to rtabmap::Transform. it's in camera's frame
                    Transform p(p_.x, p_.y, p_.z, 0, 0, 0);

                    // map coordinates of p: tmb * tbc * p.
                    // tmb is the transform from base to map, which is just the robot's
                    // pose.
                    Transform pm(pose * tbc * p);

                    /*
                    Transform t(xi, yi, zi,
                            roll, pitch, yaw);
                    Transform a = t*pm;
                    cout << (a).x() << " " << (a).y() << " " << (a).z() << endl;
                    */

                    if (idLabelMap[*it].empty())
                        idLabelMap[*it] = vector<LabelPos>();
                    LabelPos x = {detLabels[i], Point3d(pm.x(), pm.y(), pm.z()),
                            Point2d(centers[i].x, centers[i].y), areas[i]};
                    //idLabelMap[*it].push_back(LabelPos(detLabels[i], Point2d(pm.x(), pm.y())));
                    idLabelMap[*it].push_back(x);

                    if (labelIdMap[detLabels[i]].empty())
                        labelIdMap[detLabels[i]] = vector<IdPos>();
                    IdPos z = {*it, Point3d(pm.x(), pm.y(), pm.z()),
                            Point2d(centers[i].x, centers[i].y), areas[i]};
                    //labelIdMap[detLabels[i]].push_back(IdPos(*it, Point2d(pm.x(), pm.y())));
                    labelIdMap[detLabels[i]].push_back(z);

                    //cout << "testando8\n\n" << endl;
                    
                    /*
                    if (!saved)
                    {
                        std::stringstream sstm;
                        sstm << outdir;
                        if (*it < 10)
                            sstm << "00";
                        else if (*it < 100)
                            sstm << "0";
                        sstm << *it << ".jpg";
                        string fileName = sstm.str();  
                        imwrite(fileName, frame);
                        saved = true;
                    }
                    */
                }
            }

            confidences.clear();
            centers.clear();
            detLabels.clear();
            areas.clear();
        }

        it++;
    }
    std::chrono::steady_clock::time_point end_obj_det = std::chrono::steady_clock::now();
    //std::cout << "Time difference obj_det = " << 
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_obj_det - begin_obj_det).count() 
    //    << "[ms]" << std::endl;

    

    std::chrono::steady_clock::time_point begin_temp = std::chrono::steady_clock::now();
    map<int, vector<LabelPos>>::iterator it2 = idLabelMap.begin();
    int lastId = -1;
    int lastId2 = -1;
    int lastId3 = -1;

    // area threshold. if there is a change of less than aTh in the bb area, 
    // in different frames, it's probably the same object
    float aTh = 0.2; //0.3

    // real world distance threshold in meters
    //float dTh = 0.8; // 1.0
    float dTh = 7; // 1.0

    /*
    float a1 = 0.005;
    float d1 = 0.1;
    float a2 = 0.25;
    float d2 = 2.0;
    float m = (d2 - d1)/(a2 - a1);
    float b = d1 - a1*m;
    */

    // bb center threshold
    //double bbcTh = 0.0001 * (640 * 480); // 0.0001
    double bbcTh = 0.0001 * (1241 * 376); // 0.0001

    //int cnt = 0;
    //while (it2 != idLabelMap.end())
    it = ids.begin();
    contador = 0;
    while (it != ids.end())
    //while ((it != ids.end()) && (contador < 200))
    {
        // TODO
        //SensorData data = mem->getSignatureDataConst(*it);
        //data.uncompressData();


        //if ((data.laserScanRaw().data().empty()) || (data.imageRaw().empty()))
        if ((data.depthRaw().empty()) || (data.imageRaw().empty()))
        {
            it++;
            continue;
        }
        contador++;

        //int currId = it2->first;
        int currId = *it;
        //cout << "id " << currId << endl;
        //cout << "labels pos " << endl;
        //vector<LabelPos> v = it2->second;
        vector<LabelPos> v = idLabelMap[currId];
        // if first frame, all detected objects are candidates.
        //if (it == ids.begin())
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
                //cout << v[i].label << " " << v[i].cbb.x << " " << v[i].cbb.y 
                    //<< " " << v[i].area << endl;
            }
        }
        // from the second frame on, now we need to check objects association
        else
        //else if (it2 != idLabelMap.begin())
        {
            set<int> prohibitedCands;
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
                        prohibitedCands.insert(index);
                    }
                }
            }
        }
        lastId3 = lastId2;
        lastId2 = lastId;
        lastId = currId;
        //it2++;
        it++;
    }

    std::chrono::steady_clock::time_point end_temp = std::chrono::steady_clock::now();
    //std::cout << "Time difference temporal association = " << 
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_temp - begin_temp).count() 
    //    << "[ms]" << std::endl;
    //std::cout << "Time difference temporal association = " << 
    //    std::chrono::duration_cast<std::chrono::microseconds>(end_temp - begin_temp).count() 
    //    << "[us]" << std::endl;

    //cout << "000" << endl;

    //cout << "CANDOBJMAP tamanho " << candObjMap.size() << endl;
    // printa

    // now to the spatial association
    //
    std::chrono::steady_clock::time_point begin_spat = std::chrono::steady_clock::now();

    map<string, CandObj> newCandObjMap;
    candObjMap[0].label = getMaxVotedLabel(candObjMap[0]);

    //cout << "aaa" << endl;

    map<string, int> labelCount;
    bool match = false;
    int minQty = 3;
    //for (int j = 0; j < candObjMap.size(); j++)
    
    //cout << "111" << endl;

    for (auto it = candObjMap.begin(); it != candObjMap.end();)
    {
        //cout << candObjMap.size() << endl;
        //CandObj currCand = candObjMap[j];
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
    Transform t(xi, yi, zi,
            roll, pitch, yaw);

    //cout << "depois spatial" << endl;

    while (itnc != newCandObjMap.end())
    {
        //cout << "label " << itnc->first << endl;
        cout << itnc->first << " ";
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
        //cout << endl;
        //cout << obj.pm.x << " " << obj.pm.y << " " << obj.pm.z << endl;
        cout << r.x() << " " << r.y() << " " << r.z() << endl;
        itnc++;
    }

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

     
    poses.close();

    std::chrono::steady_clock::time_point end_total = std::chrono::steady_clock::now();
    //std::cout << "Time difference total = " << 
    //    std::chrono::duration_cast<std::chrono::milliseconds>(end_total - begin_total).count() 
    //    << "[ms]" << std::endl;
    return 0;
}


vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();
		
		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();
		
		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
		names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string> classes)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
	
	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
	circle(frame, Point((left+right)/2, (top+bottom)/2),4, Scalar(0,255,0),2, LINE_AA,0);	
}

void postprocess(Mat& frame, const vector<Mat>& outs, vector<float>& confidences, vector<Point2d>& centers, 
        const vector<string>& classes, 
        bool edgesValid, vector<string>& detLabels, vector<float>& areas)
{
	vector<int> classIds;
	//vector<float> confidences;
	vector<Rect> boxes;
	vector<int> frames;
	float confThreshold = 0.72; // Confidence threshold
	float nmsThreshold = 0.55;  // Non-maximum suppression threshold
	
	for (size_t i = 0; i < outs.size(); ++i)
	{
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                bool notInEdges = edgesValid ? 1 : (centerX < 0.9*frame.cols && centerX > 0.1*frame.cols 
                        && centerY < 0.9*frame.rows && centerY > 0.1*frame.rows);

                //if ( (confidence > confThreshold ) && target.compare(classes[classIdPoint.x]) == 0 
                //        && inEdges)
                if ( (confidence > confThreshold ) && notInEdges)
                {
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    //cout << "confidence: " << confidence << ". left: " << left << "top: " << top <<
            //". width: " << width << ". height: " << height << endl;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                    //detLabels.push_back(classes[classIdPoint.x]);
                }           
            }
	}

        //cout << "ANTES " << confidences.size() << endl;

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        //cout << "indices " << indices.size() << endl;
	for (size_t i = 0; i < indices.size(); ++i)
	{
            int idx = indices[i];
            //if (classes[classIds[idx]] != "cup")
            //    continue;
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                             box.x + box.width, box.y + box.height, frame, classes);
	    centers.push_back(Point2d(boxes[idx].x + boxes[idx].width/2, boxes[idx].y + boxes[idx].height/2));
            detLabels.push_back(classes[classIds[idx]]);

            //cout << detLabels[i] << endl;

            areas.push_back(boxes[idx].area());
            //cout << "label " << classes[classIds[idx]] << endl;
            //cout << "area " << boxes[idx].area() << endl;
	}

        //cout << "DEPOIS " << confidences.size() << endl;
	
        /*
	int index;
	if (confidences.size() > 0){
	  confidence =  *max_element(confidences.begin(), confidences.end());
	  index =  max_element(confidences.begin(), confidences.end()) - confidences.begin();
	  centers.push_back(Point2d(boxes[index].x + boxes[index].width/2, boxes[index].y + boxes[index].height/2));
	}else{
	  confidence = 0;
	} 
        */
}



/*
void postprocess(Mat& frame, const vector<Mat>& outs, double &confidence, Point2d &box, string target, vector<string> classes, 
        bool edgesValid)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> frames;
	float confThreshold = 0.7; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
	for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        bool inEdges = edgesValid ? 1 : (centerX < 0.8*frame.cols && centerX > 0.2*frame.cols 
                                && centerY < 0.8*frame.rows && centerY > 0.2*frame.rows);

			if ( (confidence > confThreshold ) && target.compare(classes[classIdPoint.x]) == 0 
                                && inEdges)
			//if ( (confidence > confThreshold ) && inEdges)
			{
				//int centerX = (int)(data[0] * frame.cols);
				//int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				cout << "confidence: " << confidence << ". left: " << left << "top: " << top <<
			". width: " << width << ". height: " << height << endl;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}           
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
				 box.x + box.width, box.y + box.height, frame, classes);
	}
	
	int index;
	if (confidences.size() > 0){
	  confidence =  *max_element(confidences.begin(), confidences.end());
	  index =  max_element(confidences.begin(), confidences.end()) - confidences.begin();
	  box = Point2d(boxes[index].x + boxes[index].width/2, boxes[index].y + boxes[index].height/2);
	}else{
	  confidence = 0;
	} 
}
*/