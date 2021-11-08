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
#include <numeric>

#include "Transform.h"
#include "CameraModel.h"

#include <chrono>
#include <opencv2/core/types_c.h>
#include <regex>
#include <boost/filesystem.hpp>
#include <tuple>
#include <boost/algorithm/string.hpp>


const float max_dist_for_being_static = 0.2;
const std::string obj_map_path_postfix = "/baseline_result/3d_obj_map.txt";
const std::string fv_path_postfix = "_fv_paths.txt";

struct PointXYZ { float x; float y; float z; };

struct DetObject {
    std::string label;
    PointXYZ position;
};

struct CompResult {
    std::vector<DetObject> rem_obj;
    std::vector<DetObject> new_obj;
    std::vector<DetObject> ref_static_obj;
    std::vector<DetObject> curr_static_obj;
    std::vector<DetObject> ref_displaced_obj;
    std::vector<DetObject> curr_displaced_obj;
    std::vector<std::tuple<std::string, std::string>> obj_ass; //for static and displaced objects
};

float squaredEuclideanDistance (const PointXYZ& p1, const PointXYZ& p2)
{
    float diff_x = p2.x - p1.x, diff_y = p2.y - p1.y, diff_z = p2.z - p1.z;
    return (diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
}

std::string extractSceneName(std::string path) {
    size_t last_of;
    last_of = path.find_last_of("/");

    if (last_of == path.size()-1) {
        path.erase(last_of,1);
        last_of = path.find_last_of("/");
    }
    return path.substr(last_of+1, path.size()-1);
}

std::map<std::string, std::vector<DetObject> > readObjResultFile(std::string path);
CompResult compareScenes(std::map<std::string, std::vector<DetObject> > ref_objects, std::map<std::string, std::vector<DetObject> > curr_objects, std::string ref_result_path, std::string curr_res_path);
std::vector<std::tuple<DetObject, DetObject>> computeObjSimilarity(std::vector<DetObject> ref_obj_vec, std::vector<DetObject> curr_obj_vec, std::string ref_result_path, std::string curr_result_path);
std::vector<std::vector<float>> readFeatureVectors(std::string label, std::string fv_result_path);
float computeSimilarity(std::vector<float> ref_fv_vec, std::vector<float> curr_fv_vec);

int main(int argc, char** argv)
{
    /// Parse command line arguments
    std::string room_path = argv[1];
    std::string base_result_path = argv[2];

    //extract all scene folders
    if (!boost::filesystem::exists(room_path) || !boost::filesystem::is_directory(room_path)) {
        std::cout << room_path << " does not exist or is not a directory" << std::endl;
        return -1;
    }
    std::vector<std::string> all_scene_paths;
    for (boost::filesystem::directory_entry& scene : boost::filesystem::directory_iterator(room_path)) {
        if (boost::filesystem::is_directory(scene)) {
            std::string p = scene.path().string();
            if (p.find("scene", p.length()-8) !=std::string::npos)
                all_scene_paths.push_back(p);
        }
    }
    std::sort(all_scene_paths.begin(), all_scene_paths.end());

    //----------------------------setup result folder----------------------------------
    //start at 1 because element 0 is scene1 without objects
    for (size_t idx = 1; idx < all_scene_paths.size(); idx++)
    {
        for (int k = idx + 1; k < all_scene_paths.size(); k ++)
        {
            std::string reference_path = all_scene_paths[idx];
            std::string current_path = all_scene_paths[k];
            //extract the two scene names
            std::string ref_scene_name = extractSceneName(reference_path);
            std::string curr_scene_name = extractSceneName(current_path);

            std::cout << "Comparing " << ref_scene_name << " - " << curr_scene_name << std::endl;

            std::string result_path = base_result_path + "/" + ref_scene_name + "-" + curr_scene_name + "/";
            boost::filesystem::create_directories(result_path);

            std::string ref_obj_map_path = reference_path + obj_map_path_postfix;
            std::string curr_obj_map_path = current_path + obj_map_path_postfix;

            std::map<std::string, std::vector<DetObject>> ref_objects = readObjResultFile(ref_obj_map_path);
            std::map<std::string, std::vector<DetObject>> curr_objects = readObjResultFile(curr_obj_map_path);

            CompResult result = compareScenes(ref_objects, curr_objects, boost::filesystem::path(ref_obj_map_path).parent_path().string(), boost::filesystem::path(curr_obj_map_path).parent_path().string());

            //write result to files
            if (result.rem_obj.size() > 0) {
                std::ofstream rem_obj_file;
                rem_obj_file.open(result_path + "/ref_removed_objects.txt");
                for (DetObject obj : result.rem_obj) {
                    rem_obj_file << "[" << obj.position.x << "," << obj.position.y << "," << obj.position.z << "]" << ";" << obj.label << "\n";
                }
                rem_obj_file.close();
            }
            if (result.new_obj.size() > 0) {
                std::ofstream new_obj_file;
                new_obj_file.open(result_path + "/curr_new_objects.txt");
                for (DetObject obj : result.new_obj) {
                    new_obj_file << "[" << obj.position.x << "," << obj.position.y << "," << obj.position.z << "]" << ";" << obj.label << "\n";
                }
                new_obj_file.close();
            }
            if (result.ref_displaced_obj.size() > 0) {
                std::ofstream ref_displ_obj_file;
                ref_displ_obj_file.open(result_path + "/ref_displaced_objects.txt");
                for (DetObject obj : result.ref_displaced_obj) {
                    ref_displ_obj_file << "[" << obj.position.x << "," << obj.position.y << "," << obj.position.z << "]" << ";" << obj.label << "\n";
                }
                ref_displ_obj_file.close();
            }
            if (result.curr_displaced_obj.size() > 0) {
                std::ofstream curr_displ_obj_file;
                curr_displ_obj_file.open(result_path + "/curr_displaced_objects.txt");
                for (DetObject obj : result.curr_displaced_obj) {
                    curr_displ_obj_file << "[" << obj.position.x << "," << obj.position.y << "," << obj.position.z << "]" << ";" << obj.label << "\n";
                }
                curr_displ_obj_file.close();
            }
            if (result.ref_static_obj.size() > 0) {
                std::ofstream ref_static_obj_file;
                ref_static_obj_file.open(result_path + "/ref_static_objects.txt");
                for (DetObject obj : result.ref_static_obj) {
                    ref_static_obj_file << "[" << obj.position.x << "," << obj.position.y << "," << obj.position.z << "]" << ";" << obj.label << "\n";
                }
                ref_static_obj_file.close();
            }
            if (result.curr_static_obj.size() > 0) {
                std::ofstream curr_static_obj_file;
                curr_static_obj_file.open(result_path + "/curr_static_objects.txt");
                for (DetObject obj : result.curr_static_obj) {
                    curr_static_obj_file << "[" << obj.position.x << "," << obj.position.y << "," << obj.position.z << "]" << ";" << obj.label << "\n";
                }
                curr_static_obj_file.close();
            }
            if (result.obj_ass.size() > 0) {
                std::ofstream ass_file;
                ass_file.open(result_path + "/obj_associations.txt");
                for (std::tuple<std::string, std::string> ass : result.obj_ass) {
                    ass_file << std::get<0>(ass) << ";" << std::get<1>(ass) << "\n";
                }
                ass_file.close();
            }
        }
    }
}

CompResult compareScenes(std::map<std::string, std::vector<DetObject>> ref_objects, std::map<std::string, std::vector<DetObject>> curr_objects,
                         std::string ref_result_path, std::string curr_res_path) {
    CompResult result;

    std::map<std::string, std::vector<DetObject>>::iterator ref_it;
    for (ref_it = ref_objects.begin(); ref_it != ref_objects.end(); ref_it++) {
        std::map<std::string, std::vector<DetObject>>::iterator curr_it = curr_objects.find(ref_it->first);
        if (curr_it == curr_objects.end()) {
            //no object with the same label found --> removed
            result.rem_obj.insert(result.rem_obj.end(), ref_it->second.begin(), ref_it->second.end());
            ref_it->second.clear();
        } else {
            //is there exactly one object of the class in ref and curr scene?
            if (ref_it->second.size() == 1 && curr_it->second.size() == 1) {
                //check the distance to decide if static or moved
                float squared_dist = squaredEuclideanDistance(ref_it->second[0].position, curr_it->second[0].position);
                if (std::sqrt(squared_dist) < max_dist_for_being_static) {
                    result.curr_static_obj.push_back(curr_it->second[0]);
                    result.ref_static_obj.push_back(ref_it->second[0]);
                } else {
                    result.curr_displaced_obj.push_back(curr_it->second[0]);
                    result.ref_displaced_obj.push_back(ref_it->second[0]);
                }
                result.obj_ass.push_back(std::make_tuple(ref_it->second[0].label, curr_it->second[0].label));

                auto & ref_obj_vec = ref_it->second;
                auto & curr_obj_vec = curr_it->second;
                std::string ref_label = ref_it->second[0].label;
                std::string curr_label = curr_it->second[0].label;
                ref_obj_vec.erase(std::remove_if(ref_obj_vec.begin(), ref_obj_vec.end(), [&ref_label](const DetObject &obj) { return obj.label == ref_label; }), ref_obj_vec.end());
                curr_obj_vec.erase(std::remove_if(curr_obj_vec.begin(), curr_obj_vec.end(), [&curr_label](const DetObject &obj) { return obj.label == curr_label; }), curr_obj_vec.end());
            }
            else {
                //several objects with the same class, find association via feature vectors
                std::vector<std::tuple<DetObject, DetObject>> obj_ass =  computeObjSimilarity(ref_it->second, curr_it->second, ref_result_path, curr_res_path);
                for (std::tuple<DetObject, DetObject> ass : obj_ass) {
                    float squared_dist = squaredEuclideanDistance(std::get<0>(ass).position, std::get<1>(ass).position);
                    if (std::sqrt(squared_dist) < max_dist_for_being_static) {
                        result.curr_static_obj.push_back(std::get<1>(ass));
                        result.ref_static_obj.push_back(std::get<0>(ass));
                    } else {
                        result.curr_displaced_obj.push_back(std::get<1>(ass));
                        result.ref_displaced_obj.push_back(std::get<0>(ass));
                    }
                    result.obj_ass.push_back(std::make_tuple(std::get<0>(ass).label, std::get<1>(ass).label));

                    //remove the matched objects from the maps
                    auto & ref_obj_vec = ref_it->second;
                    std::string ref_label = std::get<0>(ass).label;
                    ref_obj_vec.erase(std::remove_if(ref_obj_vec.begin(), ref_obj_vec.end(), [&ref_label](const DetObject &obj) { return obj.label == ref_label; }), ref_obj_vec.end());

                    auto & curr_obj_vec = curr_it->second;
                    std::string curr_label = std::get<1>(ass).label;
                    curr_obj_vec.erase(std::remove_if(curr_obj_vec.begin(), curr_obj_vec.end(), [&curr_label](const DetObject &obj) { return obj.label == curr_label; }), curr_obj_vec.end());
                }
            }
        }
    }

    //now handle all objects from a class where not match was found (they are either new or removed)
    for (ref_it = ref_objects.begin(); ref_it != ref_objects.end(); ref_it++) {
        result.rem_obj.insert(result.rem_obj.end(), ref_it->second.begin(), ref_it->second.end());
    }

    std::map<std::string, std::vector<DetObject>>::iterator curr_it;
    for (curr_it = curr_objects.begin(); curr_it != curr_objects.end(); curr_it++) {
        //no object with the same label found --> new
        result.new_obj.insert(result.new_obj.end(), curr_it->second.begin(), curr_it->second.end());

    }
    return result;
}

//compare each vector with each other and select the ones with minimum distance
//vectors contain all objects of a class
std::vector<std::tuple<DetObject, DetObject>> computeObjSimilarity(std::vector<DetObject> ref_obj_vec, std::vector<DetObject> curr_obj_vec,
                                                                   std::string ref_result_path, std::string curr_result_path) {
    std::vector<std::tuple<DetObject, DetObject>> obj_associations;

    std::map <std::string, std::vector<std::vector<float>>> ref_obj_fv_map;
    std::map <std::string, std::vector<std::vector<float>>> curr_obj_fv_map;

    for (DetObject ref_obj : ref_obj_vec) {
        std::vector<std::vector<float>> ref_fv = readFeatureVectors(ref_obj.label, ref_result_path);
        ref_obj_fv_map[ref_obj.label] = ref_fv;
    }
    for (DetObject curr_obj : curr_obj_vec) {
        std::vector<std::vector<float>> curr_fv = readFeatureVectors(curr_obj.label, curr_result_path);
        curr_obj_fv_map[curr_obj.label] = curr_fv;
    }

    //TODO normalize feature vectors before computing the distance, unit vector?
    std::map <std::string, std::vector<std::vector<float>>>::iterator ref_obj_it, curr_obj_it;
    for (ref_obj_it = ref_obj_fv_map.begin(); ref_obj_it != ref_obj_fv_map.end(); ref_obj_it++) {
        std::string curr_best_label="";
        float curr_best_similarity = std::numeric_limits<float>::min();
        //compare each ref fv vector against each curr fv vector of each curr object
        for (std::vector<float> ref_fv_vec : ref_obj_it->second) {
            for (curr_obj_it = curr_obj_fv_map.begin(); curr_obj_it != curr_obj_fv_map.end(); curr_obj_it++) {
                for (std::vector<float> curr_fv_vec : curr_obj_it->second) {
                    float similarity = computeSimilarity(ref_fv_vec, curr_fv_vec);
                    if (similarity > curr_best_similarity) {
                        curr_best_similarity = similarity;
                        curr_best_label = curr_obj_it->first;
                    }
                }
            }
        }
        //save the association
        DetObject ref_obj, curr_obj;
        std::string ref_label = ref_obj_it->first;
        ref_obj = *(std::find_if(ref_obj_vec.begin(), ref_obj_vec.end(), [&ref_label](const DetObject& obj) {return obj.label == ref_label;}));
        curr_obj = *(std::find_if(curr_obj_vec.begin(), curr_obj_vec.end(), [&curr_best_label](const DetObject& obj) {return obj.label == curr_best_label;}));

        obj_associations.push_back(std::make_tuple(ref_obj, curr_obj));
        curr_obj_fv_map.erase(curr_best_label);

        if (curr_obj_fv_map.size() == 0)
            return obj_associations;
    }

    return obj_associations;
}

//normalize vectors and compute dot product
float computeSimilarity(std::vector<float> ref_fv_vec, std::vector<float> curr_fv_vec) {
    if (ref_fv_vec.size() != curr_fv_vec.size()) {
        std::cerr << "[computeSimilarity]: vectors do not have the same number of elements!" << std::endl;
        exit(-1);
    }

    float ref_fv_inner = 0.0f;
    float curr_fv_inner = 0.0f;
    float ref_fv_length = std::sqrt(std::inner_product(ref_fv_vec.begin(), ref_fv_vec.end(), ref_fv_vec.begin(), ref_fv_inner));
    float curr_fv_length = std::sqrt(std::inner_product(curr_fv_vec.begin(), curr_fv_vec.end(), curr_fv_vec.begin(), curr_fv_inner));

    //norm vectors
    for (size_t i = 0; i < ref_fv_vec.size(); i++) {
        ref_fv_vec[i] = ref_fv_vec[i]/ref_fv_length;
        curr_fv_vec[i] = curr_fv_vec[i]/curr_fv_length;
    }

    float dot_product = 0.0f;
    for (int i = 0; i < ref_fv_vec.size(); i++)
        dot_product = dot_product + (ref_fv_vec[i] * curr_fv_vec[i]);
    return dot_product;
}

//read in the feature vectors from files
std::vector<std::vector<float>> readFeatureVectors(std::string label, std::string fv_result_path) {
    std::vector<std::vector<float>> ref_fv;
    std::ifstream fv_paths_ifs(fv_result_path + "/" + label + fv_path_postfix); //the files to the fv for this object are stored here
    std::string line;
    while (std::getline(fv_paths_ifs, line)){
        std::ifstream fv_ifs(line);
        if (fv_ifs) {
            std::getline(fv_ifs, line);
            line = line.substr(1, line.size() - 2); //remove first and last chars ([ and ])
            std::vector<std::string> split_result;
            boost::split(split_result, line, boost::is_any_of(", "), boost::token_compress_on);
            std::vector<float> fv_vec;
            for (std::string s : split_result)
                fv_vec.push_back(std::stof(s));

            ref_fv.push_back(fv_vec);
        }
    }
    return ref_fv;
}

std::map<std::string, std::vector<DetObject>> readObjResultFile(std::string path) {
    std::map<std::string, std::vector<DetObject>> obj_map;
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

        //remove trailing digits of the label
        label.erase(std::remove_if(std::begin(label), std::end(label),
                                   [](auto ch) { return std::isdigit(ch); }),
                    label.end());

        obj_map[label].push_back(obj);
    }
    return obj_map;
}
