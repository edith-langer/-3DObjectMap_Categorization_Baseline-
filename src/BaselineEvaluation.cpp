/**
  NOTES:
  -- if Eigen version < 3.3.6 comment line 277 and 278 in Eigen/src/Core/Matrix.h
        Base::_check_template_params();
        //if (RowsAtCompileTime!=Dynamic && ColsAtCompileTime!=Dynamic)
        //  Base::_set_noalias(other);
**/


#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <pcl/common/distances.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

typedef pcl::PointXYZRGBL PointLabel;

const float max_dist_between_objects = 0.1;
const float max_dist_for_being_static = 0.2;

float search_radius = 0.01;
float cluster_thr = 0.02;
int min_cluster_size = 200;

double ds_voxel_size = 0.002;

std::string anno_file_name = "merged_plane_clouds_ds002";

std::string c_dis_obj_name = "curr_displaced_objects.txt";
std::string c_new_obj_name = "curr_new_objects.txt";
std::string c_static_obj_name = "curr_static_objects.txt";
std::string r_dis_obj_name = "ref_displaced_objects.txt";
std::string r_rem_obj_name = "ref_removed_objects.txt";
std::string r_static_obj_name = "ref_static_objects.txt";

std::string obj_ass_name = "obj_associations.txt";


//we also need the correspondences for static/moved objects
//NEW = 0; REMOVED = 10; STATIC = [20,30,...950], MOVED = [1000, 1010,...1950]
enum ObjectClass {NEW = 0, REMOVED=1, UNKNOWN=2}; // STATIC = [20,30,...950], MOVED = [1000, 1010,...1950] - weird int assignment is better for visualization

struct SceneCompInfo {
    std::string ref_scene;
    std::string curr_scene;
    std::string result_path;
};

struct GTLabeledClouds {
    pcl::PointCloud<PointLabel>::Ptr ref_GT_cloud;
    pcl::PointCloud<PointLabel>::Ptr curr_GT_cloud;
};

struct GTObject {
    std::string name;
    pcl::PointCloud<PointLabel>::Ptr object_cloud;
    int class_label;
    GTObject(std::string _name, pcl::PointCloud<PointLabel>::Ptr _object_cloud, int _class_label=ObjectClass::UNKNOWN) :
        name(_name), object_cloud(_object_cloud), class_label(_class_label) {}

};

struct Measurements {
    int nr_det_obj = 0;
    int nr_static_obj = 0;
    int nr_novel_obj = 0;
    int nr_removed_obj = 0;
    int nr_moved_obj = 0;
};

struct GTCloudsAndNumbers {
    GTLabeledClouds clouds;
    Measurements m;
    std::vector<GTObject> ref_objects;
    std::vector<GTObject> curr_objects;
};

struct PointXYZ { float x; float y; float z; };

struct DetObject {
    std::string label;
    PointXYZ position;
};

struct AssociatedObjects {
    DetObject ref_obj;
    DetObject curr_obj;
};

float computeEuclideanDistance (PointXYZ p1, PointXYZ p2);
int findClosestObject(std::vector<DetObject> obj_vec, DetObject search_obj);
std::vector<DetObject> readObjResultFile(std::string path);
std::vector<std::tuple<std::string, std::string>> readObjAssFile(std::string path);
int findClosestObjectAssociation(std::vector<AssociatedObjects> obj_vec, AssociatedObjects search_obj);
float computeMeanPointDistance(pcl::PointCloud<PointLabel>::Ptr ref_object, pcl::PointCloud<PointLabel>::Ptr curr_obj);
void addObjectsToCloud(pcl::PointCloud<PointLabel>::Ptr cloud, std::vector<DetObject> objects);
void mergeObjectIntoMap (std::map<std::string, int> & global_map, std::string local_vec);

int getMostFrequentNumber(std::vector<int> v);
void writeSumResultsToFile(std::vector<Measurements> all_gt_results, std::vector<Measurements> all_tp_results, std::vector<Measurements> all_fp_results, std::string path);
void writeObjectSummaryToFile(std::map<std::string, int> & gt_obj_count, std::map<std::string, int> & det_obj_count, std::map<std::string, int> &tp_class_obj_count, std::vector<Measurements> &all_fp_results, std::string path);
pcl::PointCloud<PointLabel>::Ptr downsampleCloud(pcl::PointCloud<PointLabel>::Ptr input, double leafSize);
GTCloudsAndNumbers createGTforSceneComp(const std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > scene_annotations_map,
                                        const SceneCompInfo &scene_comp);

std::ostream& operator<<(std::ostream& output, Measurements const& m)
{
    output << "Detected objects: " << m.nr_det_obj << "\n";
    output << "Novel objects: " << m.nr_novel_obj << "\n";
    output << "Removed objects: " << m.nr_removed_obj << "\n";
    output << "Moved objects: " << m.nr_moved_obj << "\n";
    output << "Static objects: " << m.nr_static_obj << "\n";

    return output;
}

std::ostream& operator<<(std::ostream& output, std::vector<std::string> const& v)
{
    for (std::string s : v) {
        output << s << "; ";
    }

    return output;
}

bool readInput(std::string input_path, const pcl::PointCloud<PointLabel>::Ptr &cloud) {
    std::string ext = input_path.substr(input_path.length()-3, input_path.length());
    if (ext == "ply") {
        if (pcl::io::loadPLYFile<PointLabel> (input_path, *cloud) == -1)
        {
            std::cerr << "Couldn't read file " << input_path << std::endl;
            return false;
        }
        return true;
    } else if (ext == "pcd") {
        if (pcl::io::loadPCDFile<PointLabel> (input_path, *cloud) == -1)
        {
            std::cerr << "Couldn't read file " << input_path << std::endl;
            return false;
        }
        return true;
    } else {
        std::cerr << "Couldn't read file " << input_path << ". It needs to be a ply or pcd file" << std::endl;
        return false;
    }
}



PointXYZ computeMeanPoint(pcl::PointCloud<PointLabel>::Ptr cloud) {
    PointLabel p_mean;
    p_mean.x = p_mean.y = p_mean.z = 0.0f;

    pcl::PointCloud<PointLabel>::Ptr no_nan_cloud(new pcl::PointCloud<PointLabel>);
    std::vector<int> nan_ind;
    pcl::removeNaNFromPointCloud(*cloud, *no_nan_cloud, nan_ind);

    for (size_t i = 0; i < no_nan_cloud->size(); i++) {
        p_mean.x += no_nan_cloud->points[i].x;
        p_mean.y += no_nan_cloud->points[i].y;
        p_mean.z += no_nan_cloud->points[i].z;
    }
    p_mean.x = p_mean.x/no_nan_cloud->size();
    p_mean.y = p_mean.y/no_nan_cloud->size();
    p_mean.z = p_mean.z/no_nan_cloud->size();

    return PointXYZ{p_mean.x, p_mean.y, p_mean.z};
}

int main(int argc, char* argv[])
{
    /// Check arguments and print info
    if (argc < 3) {
        pcl::console::print_info("\n\
                                 -- Evaluation of change detection between two scenes -- : \n\
                                 \n\
                                 Syntax: %s result_folder annotation_folder \n\
                                 e.g. /home/edith/Results/Arena/ /home/edith/Annotations/Arena/",
                                 argv[0]);
        return(1);
    }

    /// Parse command line arguments
    std::string result_path = argv[1];
    std::string annotation_path = argv[2];

    if (!boost::filesystem::exists(result_path) || !boost::filesystem::is_directory(result_path)) {
        std::cerr << "Result folder does not exist " << result_path << std::endl;
    }
    if (!boost::filesystem::exists(annotation_path) || !boost::filesystem::is_directory(annotation_path)) {
        std::cerr << "Annotation folder does not exist " << annotation_path << std::endl;
    }

    //----------------------------extract all scene comparisons----------------------------------

    std::vector<SceneCompInfo> scenes;
    boost::filesystem::directory_iterator end_iter; // Default constructor for an iterator is the end iterator
    for (boost::filesystem::directory_iterator iter(result_path); iter != end_iter; ++iter) {
        if (boost::filesystem::is_directory(*iter)) {
            std::cout << iter->path() << std::endl;

            SceneCompInfo scene_info;
            scene_info.result_path = iter->path().string();
            //extract the two scene names
            std::string scene_folder = iter->path().filename().string();
            size_t last_of;
            last_of = scene_folder.find_last_of("-");
            scene_info.ref_scene = scene_folder.substr(0,last_of);
            scene_info.curr_scene = scene_folder.substr(last_of+1, scene_folder.size()-1);

            scenes.push_back(scene_info);
        }
    }


    //----------------------------read the annotation----------------------------------
    std::map<std::string, int> gt_obj_count;
    std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > scene_annotations_map; //e.g. ("scene2", ("mug", <cloud>))
    for (boost::filesystem::directory_iterator iter(annotation_path); iter != end_iter; ++iter) {
        if (boost::filesystem::is_directory(*iter)) {
            std::cout << iter->path() << std::endl;
            boost::filesystem::path planes ("planes");
            std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> objName_cloud_map;
            for (boost::filesystem::directory_iterator plane_iter(iter->path() / planes); plane_iter != end_iter; ++plane_iter) {
                std::string anno_plane_path = (plane_iter->path() / anno_file_name).string() + "_GT.anno";
                if(!boost::filesystem::exists(anno_plane_path)) {
                    std::cerr << "Couldn't find _GT.anno file for plane " << plane_iter->path().string() << ". I was looking for the file " << anno_plane_path << std::endl;
                    continue;
                }

                /// read the point cloud for the plane
                pcl::PointCloud<PointLabel>::Ptr plane_cloud(new pcl::PointCloud<PointLabel>);
                readInput((plane_iter->path() / anno_file_name).string() + ".pcd", plane_cloud);

                /// read the annotation file
                std::ifstream anno_file(anno_plane_path);
                std::string line;
                while(std::getline(anno_file, line)) {
                    pcl::PointCloud<PointLabel>::Ptr object_cloud(new pcl::PointCloud<PointLabel>);
                    std::string object_name;
                    std::vector<std::string> split_result;
                    boost::split(split_result, line, boost::is_any_of(" "));
                    object_name = split_result.at(0);
                    //special case airplane, because it consists of several parts
                    if (object_name.find("airplane") != std::string::npos) {
                        object_name = "airplane";
                    }
                    for (size_t i = 1; i < split_result.size()-1; i++) {
                        int id = std::atoi(split_result.at(i).c_str());
                        object_cloud->points.push_back(plane_cloud->points[id]);
                    }
                    object_cloud->width = object_cloud->points.size();
                    object_cloud->height = 1;
                    object_cloud->is_dense=true;
                    std::string is_on_floor = split_result[split_result.size()-1];
                    if (is_on_floor == "false") {
                        objName_cloud_map[object_name] = object_cloud;
                        std::map<std::string, int>::iterator it = gt_obj_count.find(object_name);
                        if (it == gt_obj_count.end())
                            gt_obj_count[object_name] = 1;
                        else
                            gt_obj_count[object_name] += 1;
                    }
                }
            }
            scene_annotations_map[iter->path().filename().string()] = objName_cloud_map;
        }
    }

    //----------------------------create GT for all scene comparisons from the result folder----------------------------------

    /// save scene2-scene3 (key) together with two clouds and number of detected objects
    std::map<std::string, GTCloudsAndNumbers> scene_GTClouds_map;
    for (size_t i = 0; i < scenes.size(); i++) {
        const SceneCompInfo &scene_comp = scenes[i];
        GTCloudsAndNumbers gt_cloud_numbers  = createGTforSceneComp(scene_annotations_map, scene_comp);

        if (gt_cloud_numbers.clouds.ref_GT_cloud->empty() || gt_cloud_numbers.clouds.curr_GT_cloud->empty())
            continue;

        std::string comp_string = scene_comp.ref_scene + "-" + scene_comp.curr_scene;
        scene_GTClouds_map[comp_string] = gt_cloud_numbers;
    }



    //----------------------------the real evaluation happens now----------------------------------
    /// iterate over all scene comparison folders
    std::vector<Measurements> all_gt_results, all_tp_results, all_fp_results;
    std::map<std::string, int> det_obj_count_comp, gt_obj_count_comp, tp_class_object_count;
    //measurements based on detected objects
    std::vector<Measurements> reduced_all_gt_results, reduced_all_tp_results, reduced_all_fp_results;
    std::map<std::string, int> reduced_det_obj_count_comp, reduced_gt_obj_count_comp, reduced_tp_class_object_count;
    for (size_t i = 0; i < scenes.size(); i++) {
        const SceneCompInfo &scene_comp = scenes[i];
        std::string scene_comp_str = scene_comp.ref_scene + "-" + scene_comp.curr_scene;
        const GTCloudsAndNumbers gt_cloud_numbers = scene_GTClouds_map[scene_comp_str];

        Measurements result, FP_results;
        std::vector<std::string> ref_matched_removed, curr_matched_novel, matched_moved, matched_static;
        //measurements based on detected objects
        std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > reduced_scene_annotations_map;
        Measurements reduced_result, reduced_FP_results;
        std::vector<std::string> reduced_ref_matched_removed, reduced_curr_matched_novel, reduced_matched_moved, reduced_matched_static;
        std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> curr_scene_GT_obj = scene_annotations_map[scene_comp.curr_scene];
        std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> ref_scene_GT_obj = scene_annotations_map[scene_comp.ref_scene];


        std::vector<DetObject> novel_objects = readObjResultFile(scene_comp.result_path + "/" + c_new_obj_name);
        std::vector<DetObject> removed_objects = readObjResultFile(scene_comp.result_path + "/" + r_rem_obj_name);
        std::vector<DetObject> r_moved_objects = readObjResultFile(scene_comp.result_path + "/" + r_dis_obj_name);
        std::vector<DetObject> c_moved_objects = readObjResultFile(scene_comp.result_path + "/" + c_dis_obj_name);
        std::vector<DetObject> r_static_objects = readObjResultFile(scene_comp.result_path + "/" + r_static_obj_name);
        std::vector<DetObject> c_static_objects = readObjResultFile(scene_comp.result_path + "/" + c_static_obj_name);

        //merge all GT/detected objects
        pcl::PointCloud<PointLabel>::Ptr all_GT_ref_objects_as_point(new pcl::PointCloud<PointLabel>);
        pcl::PointCloud<PointLabel>::Ptr all_det_ref_objects_as_point(new pcl::PointCloud<PointLabel>);
        pcl::PointCloud<PointLabel>::Ptr all_GT_curr_objects_as_point(new pcl::PointCloud<PointLabel>);
        pcl::PointCloud<PointLabel>::Ptr all_det_curr_objects_as_point(new pcl::PointCloud<PointLabel>);
        addObjectsToCloud(all_det_curr_objects_as_point, novel_objects);
        addObjectsToCloud(all_det_ref_objects_as_point, removed_objects);

        std::vector<DetObject> all_ref_det_objects;
        std::vector<DetObject> all_curr_det_objects;
        all_ref_det_objects = removed_objects;
        all_ref_det_objects.insert(all_ref_det_objects.end(), r_moved_objects.begin(), r_moved_objects.end());
        all_ref_det_objects.insert(all_ref_det_objects.end(), r_static_objects.begin(), r_static_objects.end());
        all_curr_det_objects = novel_objects;
        all_curr_det_objects.insert(all_curr_det_objects.end(), c_moved_objects.begin(), c_moved_objects.end());
        all_curr_det_objects.insert(all_curr_det_objects.end(), c_static_objects.begin(), c_static_objects.end());

        //create object associations
        std::vector<std::tuple<std::string, std::string>> det_object_ass_strings = readObjAssFile(scene_comp.result_path + "/" + obj_ass_name);
        std::vector<AssociatedObjects> det_static_associations, det_moved_associations;
        for (std::tuple<std::string, std::string> ass_tuple : det_object_ass_strings) {
            std::string ref_label = std::get<0>(ass_tuple);
            std::string curr_label = std::get<1>(ass_tuple);

            bool ref_is_static = false;
            bool curr_is_static =false;
            //find corresponding reference DetObject
            std::vector<DetObject>::iterator ref_it = find_if(r_moved_objects.begin(), r_moved_objects.end(),
                                                              [ref_label](const DetObject ref_obj){return ref_obj.label == ref_label;});
            //if not found in  moved objects, then it should be in static objects
            if (ref_it == r_moved_objects.end()) {
                ref_it = find_if(r_static_objects.begin(), r_static_objects.end(),
                                 [ref_label](const DetObject ref_obj){return ref_obj.label == ref_label;});
                ref_is_static=true;
                if (ref_it == r_static_objects.end()) {
                    std::cerr << "Could not find the following object from reference scene either in displaced nor static result files: " << ref_label << std::endl;
                    exit(-1);
                }
            }

            //find corresponding current DetObject
            std::vector<DetObject>::iterator curr_it = find_if(c_moved_objects.begin(), c_moved_objects.end(),
                                                               [curr_label](const DetObject curr_obj){return curr_obj.label == curr_label;});
            //if not found in  moved objects, then it should be in static objects
            if (curr_it == c_moved_objects.end()) {
                curr_it = find_if(c_static_objects.begin(), c_static_objects.end(),
                                  [curr_label](const DetObject curr_obj){return curr_obj.label == curr_label;});
                curr_is_static = true;
                if (curr_it == c_static_objects.end()) {
                    std::cerr << "Could not find the following object from current scene either in displaced nor static result files: " << curr_label << std::endl;
                    exit(-1);
                }
            }

            if (curr_is_static != ref_is_static) {
                std::cerr << "Found object association where one object is static while the other is moved: " << ref_label << "-" << curr_label << std::endl;
                exit(-1);
            }
            if (ref_is_static)
                det_static_associations.push_back(AssociatedObjects{*ref_it, *curr_it});
            else
                det_moved_associations.push_back(AssociatedObjects{*ref_it, *curr_it});

            addObjectsToCloud(all_det_ref_objects_as_point, {*ref_it});
            addObjectsToCloud(all_det_curr_objects_as_point, {*curr_it});
        }


        //number of detected objects = GT objects which have a detected object in their visinity
        for (GTObject gt_obj : gt_cloud_numbers.ref_objects) {
            mergeObjectIntoMap(gt_obj_count_comp, gt_obj.name);
            PointXYZ center_point = computeMeanPoint(gt_obj.object_cloud);
            DetObject obj {gt_obj.name, center_point};
            int closest_idx = findClosestObject(all_ref_det_objects, obj);
            if (closest_idx >= 0) {
                result.nr_det_obj += 1;
                mergeObjectIntoMap(det_obj_count_comp, gt_obj.name);
            }
            else {
                //remove it from GT for reduced set
                ref_scene_GT_obj.erase(gt_obj.name);
            }
        }
        for (GTObject gt_obj : gt_cloud_numbers.curr_objects) {
            mergeObjectIntoMap(gt_obj_count_comp, gt_obj.name);
            PointXYZ center_point = computeMeanPoint(gt_obj.object_cloud);
            DetObject obj {gt_obj.name, center_point};
            int closest_idx = findClosestObject(all_curr_det_objects, obj);
            if (closest_idx >= 0) {
                result.nr_det_obj += 1;
                mergeObjectIntoMap(det_obj_count_comp, gt_obj.name);
            }
            else {
                //remove it from GT for reduced set
                curr_scene_GT_obj.erase(gt_obj.name);
            }
        }

        //create GT based on detected objects
        reduced_scene_annotations_map[scene_comp.curr_scene] = curr_scene_GT_obj;
        reduced_scene_annotations_map[scene_comp.ref_scene] = ref_scene_GT_obj;

        const GTCloudsAndNumbers reduced_gt_cloud_numbers  = createGTforSceneComp(reduced_scene_annotations_map, scene_comp);


        //----------CHECK NOVEL OBJECTS------------------------------
        //extract GT new objects and compute a center point
        std::vector<DetObject> gt_new_objects;
        for (const GTObject gt_new_obj : gt_cloud_numbers.curr_objects) {
            if (gt_new_obj.class_label == ObjectClass::NEW) {
                PointXYZ center_point = computeMeanPoint(gt_new_obj.object_cloud);
                DetObject obj {gt_new_obj.name, center_point};
                gt_new_objects.push_back(obj);
            }
        }
        addObjectsToCloud(all_GT_curr_objects_as_point, gt_new_objects);


        //for each detected new object, check if there is a new GT object close by
        std::vector<DetObject> rem_new_det_obj;
        for (DetObject det_obj : novel_objects) {
            //find the closest GT and check the distance
            int closest_idx = findClosestObject(gt_new_objects, det_obj);
            if (closest_idx >= 0) {
                result.nr_novel_obj++;
                mergeObjectIntoMap(tp_class_object_count, gt_new_objects[closest_idx].label);
                gt_new_objects.erase(gt_new_objects.begin() + closest_idx);
            } else {
                rem_new_det_obj.push_back(det_obj);
            }
        }
        FP_results.nr_novel_obj += rem_new_det_obj.size();

        //-----------------------------------------------------------

        //----------CHECK REMOVED OBJECTS------------------------------
        //extract GT removed objects and compute a center point
        std::vector<DetObject> gt_removed_objects;
        for (const GTObject gt_removed_obj : gt_cloud_numbers.ref_objects) {
            if (gt_removed_obj.class_label == ObjectClass::REMOVED) {
                PointXYZ center_point = computeMeanPoint(gt_removed_obj.object_cloud);
                DetObject obj {gt_removed_obj.name, center_point};
                gt_removed_objects.push_back(obj);
            }
        }
        addObjectsToCloud(all_GT_ref_objects_as_point, gt_removed_objects);

        //for each detected new object, check if there is a new GT object close by
        std::vector<DetObject> rem_removed_det_obj;
        for (DetObject det_obj : removed_objects) {
            //find the closest GT and check the distance
            int closest_idx = findClosestObject(gt_removed_objects, det_obj);
            if (closest_idx >= 0) {
                result.nr_removed_obj++;
                mergeObjectIntoMap(tp_class_object_count, gt_removed_objects[closest_idx].label);
                gt_removed_objects.erase(gt_removed_objects.begin() + closest_idx);
            } else {
                rem_removed_det_obj.push_back(det_obj);
            }
        }
        FP_results.nr_removed_obj += rem_removed_det_obj.size();

        //-----------------------------------------------------------


        //----------CHECK DISPLACED OBJECTS------------------------------
        //extract GT new objects and compute a center point
        std::vector<AssociatedObjects> gt_displaced_objects;
        for (const GTObject gt_ref_displaced_obj : gt_cloud_numbers.ref_objects) {
            if (gt_ref_displaced_obj.class_label >= 1000 && gt_ref_displaced_obj.class_label <= 1950) { //DISPLACED
                /// find the corresponding curr object
                std::vector<GTObject>::const_iterator curr_it = find_if(gt_cloud_numbers.curr_objects.begin(), gt_cloud_numbers.curr_objects.end(),
                                                                        [gt_ref_displaced_obj](const GTObject curr_obj){return gt_ref_displaced_obj.class_label == curr_obj.class_label;});
                PointXYZ ref_center_point = computeMeanPoint(gt_ref_displaced_obj.object_cloud);
                DetObject obj_ref {gt_ref_displaced_obj.name, ref_center_point};
                PointXYZ curr_center_point = computeMeanPoint(curr_it->object_cloud);
                DetObject obj_curr {curr_it->name, curr_center_point};
                AssociatedObjects ass_obj {obj_ref, obj_curr};
                gt_displaced_objects.push_back(ass_obj);

                addObjectsToCloud(all_GT_ref_objects_as_point, {obj_ref});
                addObjectsToCloud(all_GT_curr_objects_as_point, {obj_curr});
            }
        }

        //for each detected static object pair, check if there is a static GT object pair close by
        std::vector<DetObject> rem_ref_displaced_det_obj, rem_curr_displaced_det_obj;
        for (AssociatedObjects det_obj : det_moved_associations) {
            //find the closest GT and check the distance
            int closest_idx = findClosestObjectAssociation(gt_displaced_objects, det_obj);
            if (closest_idx >= 0) {
                result.nr_moved_obj += 2;
                mergeObjectIntoMap(tp_class_object_count, gt_displaced_objects[closest_idx].ref_obj.label);
                mergeObjectIntoMap(tp_class_object_count, gt_displaced_objects[closest_idx].curr_obj.label);
                gt_displaced_objects.erase(gt_displaced_objects.begin() + closest_idx);
            } else {
                rem_ref_displaced_det_obj.push_back(det_obj.ref_obj);
                rem_curr_displaced_det_obj.push_back(det_obj.curr_obj);
            }
        }
        FP_results.nr_moved_obj += rem_ref_displaced_det_obj.size();
        FP_results.nr_moved_obj += rem_curr_displaced_det_obj.size();

        //-----------------------------------------------------------


        //----------CHECK STATIC OBJECTS------------------------------
        //extract GT new objects and compute a center point
        std::vector<AssociatedObjects> gt_static_objects;
        for (const GTObject gt_ref_static_obj : gt_cloud_numbers.ref_objects) {
            if (gt_ref_static_obj.class_label >= 20 && gt_ref_static_obj.class_label <= 950) { //STATIC
                /// find the corresponding curr object
                std::vector<GTObject>::const_iterator curr_it = find_if(gt_cloud_numbers.curr_objects.begin(), gt_cloud_numbers.curr_objects.end(),
                                                                        [gt_ref_static_obj](const GTObject curr_obj){return gt_ref_static_obj.class_label == curr_obj.class_label;});
                PointXYZ ref_center_point = computeMeanPoint(gt_ref_static_obj.object_cloud);
                DetObject obj_ref {gt_ref_static_obj.name, ref_center_point};
                PointXYZ curr_center_point = computeMeanPoint(curr_it->object_cloud);
                DetObject obj_curr {curr_it->name, curr_center_point};
                AssociatedObjects ass_obj {obj_ref, obj_curr};
                gt_static_objects.push_back(ass_obj);

                addObjectsToCloud(all_GT_ref_objects_as_point, {obj_ref});
                addObjectsToCloud(all_GT_curr_objects_as_point, {obj_curr});
            }
        }

        //for each detected static object pair, check if there is a static GT object pair close by
        std::vector<DetObject> rem_ref_static_det_obj, rem_curr_static_det_obj;
        for (AssociatedObjects det_obj : det_static_associations) {
            //find the closest GT and check the distance
            int closest_idx = findClosestObjectAssociation(gt_static_objects, det_obj);
            if (closest_idx >= 0) {
                result.nr_static_obj += 2;
                mergeObjectIntoMap(tp_class_object_count, gt_static_objects[closest_idx].ref_obj.label);
                mergeObjectIntoMap(tp_class_object_count, gt_static_objects[closest_idx].curr_obj.label);
                gt_static_objects.erase(gt_static_objects.begin() + closest_idx);
            } else {
                rem_ref_static_det_obj.push_back(det_obj.ref_obj);
                rem_curr_static_det_obj.push_back(det_obj.curr_obj);
            }
        }

        //special case for static objects - remaining objects are not necesarely FPs, but background objects that were detected
        //for all remaining (not matched GT objects) check if there is one close by in the not matched static objects
        std::vector<DetObject> ref_all_remaining_GT_objects, curr_all_remaining_GT_objects;
        ref_all_remaining_GT_objects.insert(ref_all_remaining_GT_objects.begin(), gt_removed_objects.begin(), gt_removed_objects.end());
        curr_all_remaining_GT_objects.insert(curr_all_remaining_GT_objects.begin(), gt_new_objects.begin(), gt_new_objects.end());
        for (AssociatedObjects ass_obj : gt_displaced_objects) {
            ref_all_remaining_GT_objects.push_back(ass_obj.ref_obj);
            curr_all_remaining_GT_objects.push_back(ass_obj.curr_obj);
        }
        for (DetObject det_obj : rem_ref_static_det_obj) {
            //find the closest GT and check the distance
            int closest_idx = findClosestObject(ref_all_remaining_GT_objects, det_obj);
            if (closest_idx >= 0) {
                FP_results.nr_static_obj += 1;
                ref_all_remaining_GT_objects.erase(ref_all_remaining_GT_objects.begin() + closest_idx);
            }
        }
        for (DetObject det_obj : rem_curr_static_det_obj) {
            //find the closest GT and check the distance
            int closest_idx = findClosestObject(curr_all_remaining_GT_objects, det_obj);
            if (closest_idx >= 0) {
                FP_results.nr_static_obj += 1;
                curr_all_remaining_GT_objects.erase(curr_all_remaining_GT_objects.begin() + closest_idx);
            }
        }

        pcl::io::savePCDFileBinary(scene_comp.result_path + "/all_GT_ref_objects_as_points.pcd", *all_GT_ref_objects_as_point);
        pcl::io::savePCDFileBinary(scene_comp.result_path + "/all_GT_curr_objects_as_points.pcd", *all_GT_curr_objects_as_point);
        pcl::io::savePCDFileBinary(scene_comp.result_path + "/all_det_ref_objects_as_points.pcd", *all_det_ref_objects_as_point);
        pcl::io::savePCDFileBinary(scene_comp.result_path + "/all_det_curr_objects_as_points.pcd", *all_det_curr_objects_as_point);

        std::cout << scene_comp.ref_scene + "-" + scene_comp.curr_scene << std::endl;
        std::cout << "GT numbers \n" << gt_cloud_numbers.m;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "True Positives" << "\n" << result;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "False Positives" << "\n" << FP_results;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Found NOVEL objects: " << curr_matched_novel << std::endl;
        std::cout << "Found REMOVED objects: " << ref_matched_removed << std::endl;
        std::cout << "Found MOVED objects: " << matched_moved << std::endl;
        std::cout << "Found STATIC objects: " << matched_static << std::endl;
        std::cout << "###############################################" << std::endl;

        std::ofstream scene_comp_result_file;
        scene_comp_result_file.open(scene_comp.result_path+"/result.txt");
        scene_comp_result_file << "GT numbers \n" << gt_cloud_numbers.m;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "True Positives" << "\n" << result;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "False Positives" << "\n" << FP_results;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "Found NOVEL objects: " << curr_matched_novel << "\n";
        scene_comp_result_file << "Found REMOVED objects: " << ref_matched_removed << "\n";
        scene_comp_result_file << "Found MOVED objects: " << matched_moved << "\n";
        scene_comp_result_file << "Found STATIC objects: " << matched_static << "\n";

        //        scene_comp_result_file << "\n-------------------------------Based on detected objects----------------------------------------\n";
        //        scene_comp_result_file << "GT numbers \n" << reduced_gt_cloud_numbers.m;
        //        scene_comp_result_file << "-------------------------------------------" << "\n";
        //        scene_comp_result_file << "True Positives" << "\n" << reduced_result;
        //        scene_comp_result_file << "-------------------------------------------" << "\n";
        //        scene_comp_result_file << "False Positives" << "\n" << reduced_FP_results;
        //        scene_comp_result_file << "-------------------------------------------" << "\n";
        //        scene_comp_result_file << "Found NOVEL objects: " << reduced_curr_matched_novel << "\n";
        //        scene_comp_result_file << "Found REMOVED objects: " << reduced_ref_matched_removed << "\n";
        //        scene_comp_result_file << "Found MOVED objects: " << reduced_matched_moved << "\n";
        //        scene_comp_result_file << "Found STATIC objects: " << reduced_matched_static << "\n";
        scene_comp_result_file.close();

        all_gt_results.push_back(gt_cloud_numbers.m);
        all_tp_results.push_back(result);
        all_fp_results.push_back(FP_results);

        //        reduced_all_gt_results.push_back(reduced_gt_cloud_numbers.m);
        //        reduced_all_tp_results.push_back(reduced_result);
        //        reduced_all_fp_results.push_back(reduced_FP_results);

    }

    std::cout << "Overview of objects used in the scenes" << std::endl; //sum of objects used in scene2-scene6
    for (const auto obj_count : gt_obj_count) {
        std::cout << obj_count.first << ": " << obj_count.second << std::endl;
    }

    std::cout << "Number of GT objects using all possible scene comparisons" << std::endl; //sum of objects used in scene2-scene6
    for (const auto obj_count : gt_obj_count_comp) {
        std::cout << obj_count.first << ": " << obj_count.second << std::endl;
    }

    std::cout << "Number of detected objects using all possible scene comparisons" << std::endl; //sum of objects used in scene2-scene6
    for (const auto obj_count : det_obj_count_comp) {
        std::cout << obj_count.first << ": " << obj_count.second << std::endl;
    }

    std::string result_file = result_path +  "/results.txt";
    //clear the file content
    std::ofstream result_stream (result_file);
    result_stream.close();
    writeSumResultsToFile(all_gt_results, all_tp_results, all_fp_results, result_file);
    writeObjectSummaryToFile(gt_obj_count_comp, det_obj_count_comp, tp_class_object_count, all_fp_results, result_file);

    //write results based on detected objects
    //    result_stream.open (result_file, std::ofstream::out | std::ofstream::app);
    //    result_stream << "\n\n-------------------------------Based on detected objects----------------------------------------\n";
    //    result_stream.close();
    //    writeSumResultsToFile(reduced_all_gt_results, reduced_all_tp_results, reduced_all_fp_results, result_file);
    //    writeObjectSummaryToFile(reduced_gt_obj_count_comp, reduced_det_obj_count_comp, reduced_tp_class_object_count, reduced_all_fp_results, result_file);

}


void writeObjectSummaryToFile(std::map<std::string, int> & gt_obj_count, std::map<std::string, int> & det_obj_count, std::map<std::string, int> & tp_class_obj_count,
                              std::vector<Measurements> &all_fp_results, std::string path) {
    int count_gt_obj = 0;
    int count_det_obj = 0;
    int count_tp_class_obj = 0;
    std::ofstream result_file;
    result_file.open (path, std::ofstream::out | std::ofstream::app);
    result_file << "\n";
    for (const auto obj_count : gt_obj_count) {
        const std::string &obj_name = obj_count.first;
        result_file << obj_count.first << ": " << obj_count.second;
        count_gt_obj += obj_count.second;
        std::map<std::string, int>::iterator det_it = det_obj_count.find(obj_name);
        if (det_it != det_obj_count.end()) {
            result_file << "/" << det_it->second;
            count_det_obj += det_it->second;
        } else {
            result_file << "/" << 0;
        }
        det_it = tp_class_obj_count.find(obj_name);
        if (det_it != tp_class_obj_count.end()) {
            result_file << "/" << det_it->second;
            count_tp_class_obj += det_it->second;
        } else {
            result_file << "/" << 0;
        }
        result_file << "\n";
    }

    //sum of FP
    int FP_sum=0;
    for (Measurements m : all_fp_results) {
        FP_sum += m.nr_moved_obj;
        FP_sum += m.nr_novel_obj;
        FP_sum += m.nr_removed_obj;
        FP_sum += m.nr_static_obj;
    }

    result_file << "#objects in scene: " << count_gt_obj << "\n";
    result_file << "#detected objects: " << count_det_obj << "\n";
    result_file << "#objects correctly classified: " << count_tp_class_obj << "\n";
    result_file << "#objects wrongly classified: " << FP_sum << "\n";
}

void writeSumResultsToFile(std::vector<Measurements> all_gt_results, std::vector<Measurements> all_tp_results, std::vector<Measurements> all_fp_results,
                           std::string path) {
    Measurements gt_sum, tp_sum, fp_sum;
    for (Measurements m : all_gt_results) {
        gt_sum.nr_det_obj += m.nr_det_obj;
        gt_sum.nr_moved_obj += m.nr_moved_obj;
        gt_sum.nr_novel_obj += m.nr_novel_obj;
        gt_sum.nr_removed_obj += m.nr_removed_obj;
        gt_sum.nr_static_obj += m.nr_static_obj;
    }
    for (Measurements m : all_tp_results) {
        tp_sum.nr_det_obj += m.nr_det_obj;
        tp_sum.nr_moved_obj += m.nr_moved_obj;
        tp_sum.nr_novel_obj += m.nr_novel_obj;
        tp_sum.nr_removed_obj += m.nr_removed_obj;
        tp_sum.nr_static_obj += m.nr_static_obj;
    }
    for (Measurements m : all_fp_results) {
        fp_sum.nr_det_obj += m.nr_det_obj;
        fp_sum.nr_moved_obj += m.nr_moved_obj;
        fp_sum.nr_novel_obj += m.nr_novel_obj;
        fp_sum.nr_removed_obj += m.nr_removed_obj;
        fp_sum.nr_static_obj += m.nr_static_obj;
    }

    std::ofstream result_file;
    result_file.open (path, std::ofstream::out | std::ofstream::app);
    result_file << "Ground Truth \n" << gt_sum;
    result_file << "\nTrue Positives \n" << tp_sum;
    result_file << "\nFalse Positives \n" << fp_sum;
    result_file.close();
}

int getMostFrequentNumber(std::vector<int> v) {
    int maxCount = 0, mostElement = *(v.begin());
    int sz = v.size(); // to avoid calculating the size every time
    for(int i=0; i < sz; i++)
    {
        int c = count(v.begin(), v.end(), v.at(i));
        if(c > maxCount)
        {   maxCount = c;
            mostElement = v.at(i);
        }
    }
    return mostElement;
}

pcl::PointCloud<PointLabel>::Ptr downsampleCloud(pcl::PointCloud<PointLabel>::Ptr input, double leafSize)
{
    std::cout << "PointCloud before filtering has: " << input->points.size () << " data points." << std::endl;

    // Create the filtering object: downsample the dataset using a leaf size
    pcl::VoxelGrid<PointLabel> vg;
    pcl::PointCloud<PointLabel>::Ptr cloud_filtered (new pcl::PointCloud<PointLabel>);
    vg.setInputCloud (input);
    vg.setLeafSize (leafSize, leafSize, leafSize);
    vg.setDownsampleAllData(true);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;

    return cloud_filtered;
}

GTCloudsAndNumbers createGTforSceneComp(const std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > scene_annotations_map,
                                        const SceneCompInfo &scene_comp) {
    GTCloudsAndNumbers gt_cloud_numbers;
    //get the corresponding annotated clouds
    std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> >::const_iterator scene_anno_it_ref = scene_annotations_map.find(scene_comp.ref_scene);
    if (scene_anno_it_ref == scene_annotations_map.end()) {
        std::cerr << "Couldn't find the GT in the map for scene " << scene_comp.ref_scene << std::endl;
        return gt_cloud_numbers;
    }
    std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> >::const_iterator scene_anno_it_curr = scene_annotations_map.find(scene_comp.curr_scene);
    if (scene_anno_it_curr == scene_annotations_map.end()) {
        std::cerr << "Couldn't find the GT in the map for scene " << scene_comp.curr_scene << std::endl;
        return gt_cloud_numbers;
    }

    const std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> &ref_objName_cloud_map = scene_anno_it_ref->second;
    const std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> &curr_objName_cloud_map = scene_anno_it_curr->second;

    //merge all objects in a labeled cloud depending if it is new/removed/moved/static
    pcl::PointCloud<PointLabel>::Ptr ref_GT_cloud(new pcl::PointCloud<PointLabel>);
    pcl::PointCloud<PointLabel>::Ptr curr_GT_cloud(new pcl::PointCloud<PointLabel>);

    Measurements gt_measurements;
    std::vector<GTObject> ref_gt_objects;
    std::vector<GTObject> curr_gt_objects;

    uint static_cnt = 20;
    uint moved_cnt = 1000;
    // go through all objects of the ref scene and classify them
    std::map<std::string, pcl::PointCloud<PointLabel>::Ptr>::const_iterator ref_map_it;
    for (ref_map_it = ref_objName_cloud_map.begin(); ref_map_it != ref_objName_cloud_map.end(); ref_map_it++) {
        const std::string &ref_obj_name = ref_map_it->first;
        std::map<std::string, pcl::PointCloud<PointLabel>::Ptr>::const_iterator curr_map_it = curr_objName_cloud_map.find(ref_obj_name);
        if (curr_map_it == curr_objName_cloud_map.end()) { //ref object removed
            for (size_t i = 0; i < ref_map_it->second->points.size(); i++) {
                ref_map_it->second->points[i].label = ObjectClass::REMOVED;
            }
            gt_measurements.nr_removed_obj++;
            ref_gt_objects.push_back(GTObject(ref_obj_name, ref_map_it->second, ObjectClass::REMOVED));
        } else { //check if it is either static or moved. label also the corresponding curr objects
            float obj_dist = computeMeanPointDistance(ref_map_it->second, curr_map_it->second);
            if (obj_dist < max_dist_for_being_static) {
                for (size_t i = 0; i < ref_map_it->second->points.size(); i++) {
                    ref_map_it->second->points[i].label = static_cnt; //ObjectClass::STATIC;
                }
                for (size_t i = 0; i < curr_map_it->second->points.size(); i++) {
                    curr_map_it->second->points[i].label = static_cnt; //ObjectClass::STATIC;
                }
                ref_gt_objects.push_back(GTObject(ref_obj_name, ref_map_it->second, static_cnt));
                curr_gt_objects.push_back(GTObject(curr_map_it->first, curr_map_it->second, static_cnt));
                gt_measurements.nr_static_obj+=2;
                static_cnt += 10;
            } else {
                for (size_t i = 0; i < ref_map_it->second->points.size(); i++) {
                    ref_map_it->second->points[i].label = moved_cnt; //ObjectClass::MOVED;
                }
                for (size_t i = 0; i < curr_map_it->second->points.size(); i++) {
                    curr_map_it->second->points[i].label = moved_cnt; //ObjectClass::MOVED;
                }
                ref_gt_objects.push_back(GTObject(ref_obj_name, ref_map_it->second, moved_cnt));
                curr_gt_objects.push_back(GTObject(curr_map_it->first, curr_map_it->second, moved_cnt));
                moved_cnt += 10;
                gt_measurements.nr_moved_obj+=2;
            }
            *curr_GT_cloud += *(curr_map_it->second);
        }
        *ref_GT_cloud += *(ref_map_it->second);
    }


    // go through all objects of the curr scene and classify them
    std::map<std::string, pcl::PointCloud<PointLabel>::Ptr>::const_iterator curr_map_it;
    for (curr_map_it = curr_objName_cloud_map.begin(); curr_map_it != curr_objName_cloud_map.end(); curr_map_it++) {
        const std::string &curr_obj_name = curr_map_it->first;
        if (ref_objName_cloud_map.find(curr_obj_name) == ref_objName_cloud_map.end()) { //curr object new
            for (size_t i = 0; i < curr_map_it->second->points.size(); i++) {
                curr_map_it->second->points[i].label = ObjectClass::NEW;
            }
            curr_gt_objects.push_back(GTObject(curr_map_it->first, curr_map_it->second, ObjectClass::NEW));
            gt_measurements.nr_novel_obj++;
            *curr_GT_cloud += *(curr_map_it->second);
        }
    }

    gt_measurements.nr_det_obj = gt_measurements.nr_novel_obj + gt_measurements.nr_removed_obj + gt_measurements.nr_static_obj + gt_measurements.nr_moved_obj;

    pcl::io::savePCDFileBinary(scene_comp.result_path + "/" + scene_comp.ref_scene + "-" + scene_comp.curr_scene + ".pcd", *ref_GT_cloud);
    pcl::io::savePCDFileBinary(scene_comp.result_path + "/" + scene_comp.curr_scene + "-" + scene_comp.ref_scene + ".pcd", *curr_GT_cloud);

    GTLabeledClouds gt_labeled_clouds;
    gt_labeled_clouds.ref_GT_cloud = ref_GT_cloud;
    gt_labeled_clouds.curr_GT_cloud = curr_GT_cloud;

    //we downsample the GT clouds accordingly to the leaf size used to create the results
    gt_labeled_clouds.ref_GT_cloud = downsampleCloud(gt_labeled_clouds.ref_GT_cloud, ds_voxel_size);
    gt_labeled_clouds.curr_GT_cloud = downsampleCloud(gt_labeled_clouds.curr_GT_cloud, ds_voxel_size);
    for (GTObject& o : ref_gt_objects)
        o.object_cloud = downsampleCloud(o.object_cloud, ds_voxel_size);
    for (GTObject& o : curr_gt_objects)
        o.object_cloud = downsampleCloud(o.object_cloud, ds_voxel_size);


    gt_cloud_numbers.clouds = gt_labeled_clouds;
    gt_cloud_numbers.m = gt_measurements;
    gt_cloud_numbers.ref_objects = ref_gt_objects;
    gt_cloud_numbers.curr_objects = curr_gt_objects;

    return gt_cloud_numbers;

}


std::vector<DetObject> readObjResultFile(std::string path) {
    std::vector<DetObject> obj_map;
    std::ifstream ifs(path.c_str());
    std::string line;
    while (std::getline(ifs, line)){
        //parse [-1.28064,-0.376684,0.818286];apple0
        std::vector<std::string> split_result;
        boost::split(split_result, line, boost::is_any_of(";"), boost::token_compress_on);
        std::string label = split_result[1];
        std::string coord_str = split_result[0];
        coord_str = coord_str.substr(1, coord_str.size() - 2); //remove first and last chars ([ and ])
        boost::split(split_result, coord_str, boost::is_any_of(","), boost::token_compress_on);

        DetObject obj{label, PointXYZ{std::stof(split_result[0]), std::stof(split_result[1]), std::stof(split_result[2])}};

        obj_map.push_back(obj);
    }
    return obj_map;
}

std::vector<std::tuple<std::string, std::string>> readObjAssFile(std::string path) {
    std::vector<std::tuple<std::string, std::string>> obj_associations;
    std::ifstream ifs(path.c_str());
    std::string line;
    while (std::getline(ifs, line)){
        //parse book1;book0
        std::vector<std::string> split_result;
        boost::split(split_result, line, boost::is_any_of(";"), boost::token_compress_on);
        std::string ref_obj_label = split_result[0];
        std::string curr_obj_label = split_result[1];

        obj_associations.push_back(std::make_tuple(ref_obj_label, curr_obj_label));
    }
    return obj_associations;
}


//compute euclidean distance between search point and all point in a vector
//returns -1 if no point is closer than max_dist_between_objects, otherwise the vector index of the closest element
int findClosestObject(std::vector<DetObject> obj_vec, DetObject search_obj) {
    int closest_ind = -1;
    float min_dist = max_dist_between_objects;

    for (size_t i = 0; i < obj_vec.size(); i++) {
        const DetObject &obj = obj_vec[i];
        float dist = computeEuclideanDistance(obj.position, search_obj.position);
        if (dist < min_dist) {
            min_dist = dist;
            closest_ind = i;
        }
    }
    return closest_ind;
}

//compute euclidean distance between two object associations
//returns -1 if no point is closer than max_dist_between_objects, otherwise the vector index of the closest element
int findClosestObjectAssociation(std::vector<AssociatedObjects> obj_vec, AssociatedObjects search_obj) {
    int closest_ind = -1;
    float min_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < obj_vec.size(); i++) {
        const AssociatedObjects &obj = obj_vec[i];
        float ref_dist = computeEuclideanDistance(obj.ref_obj.position, search_obj.ref_obj.position);
        float curr_dist = computeEuclideanDistance(obj.curr_obj.position, search_obj.curr_obj.position);
        if (ref_dist < max_dist_between_objects && curr_dist < max_dist_between_objects ) {
            if ((ref_dist + curr_dist) < min_dist) {
                min_dist = (ref_dist + curr_dist);
                closest_ind = i;
            }
        }
    }
    return closest_ind;
}

float computeEuclideanDistance (PointXYZ p1, PointXYZ p2) {
    float diff_x = p1.x - p2.x, diff_y = p1.y - p2.y, diff_z = p1.z - p2.z;
    float dist = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

    return dist;
}
float computeMeanPointDistance(pcl::PointCloud<PointLabel>::Ptr ref_object, pcl::PointCloud<PointLabel>::Ptr curr_obj) {
    PointLabel ref_p_mean, curr_p_mean;
    ref_p_mean.x = ref_p_mean.y = ref_p_mean.z = 0.0f;
    curr_p_mean = ref_p_mean;

    for (size_t i = 0; i < ref_object->size(); i++) {
        ref_p_mean.x += ref_object->points[i].x;
        ref_p_mean.y += ref_object->points[i].y;
        ref_p_mean.z += ref_object->points[i].z;
    }
    ref_p_mean.x = ref_p_mean.x/ref_object->size();
    ref_p_mean.y = ref_p_mean.y/ref_object->size();
    ref_p_mean.z = ref_p_mean.z/ref_object->size();

    for (size_t i = 0; i < curr_obj->size(); i++) {
        curr_p_mean.x += curr_obj->points[i].x;
        curr_p_mean.y += curr_obj->points[i].y;
        curr_p_mean.z += curr_obj->points[i].z;
    }
    curr_p_mean.x = curr_p_mean.x/curr_obj->size();
    curr_p_mean.y = curr_p_mean.y/curr_obj->size();
    curr_p_mean.z = curr_p_mean.z/curr_obj->size();

    return pcl::euclideanDistance(ref_p_mean, curr_p_mean);
}

void addObjectsToCloud(pcl::PointCloud<PointLabel>::Ptr cloud, std::vector<DetObject> objects) {
    for (DetObject obj : objects) {
        PointLabel p;
        p.x = obj.position.x;
        p.y = obj.position.y;
        p.z = obj.position.z;
        cloud->points.push_back(p);
    }
}

void mergeObjectIntoMap (std::map<std::string, int> & global_map, std::string local_vec) {
    std::map<std::string, int>::iterator it = global_map.find(local_vec);
    if (it == global_map.end())
        global_map[local_vec] = 1;
    else
        global_map[local_vec] += 1;
}

