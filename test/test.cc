#include <fstream>
#include <vector>
#include <string>
#include "utils.h"
#include "lsm_vec_db.h"
#include "lsm_vec_index.h"
#include "config.h"



int main(int argc, char* argv[])
{
    Config config = Config::Parse(argc, argv);

    std::ofstream outFile(config.output_path);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << config.output_path << "\n";
        return 1;
    }

    int vectorDim = getdim(config.input_file_path);
    std::cout << "Vector dimension: " << vectorDim << "\n";

    lsm_vec::LSMVecDBOptions options;
    options.dim = vectorDim;
    options.m = config.M;
    options.m_max = config.Mmax;
    options.m_level = config.Ml;
    options.ef_construction = config.efConstruction;
    options.vec_file_capacity = config.vec_file_capacity;
    options.paged_max_cached_pages = config.paged_max_cached_pages;
    options.vector_storage_type = config.vector_storage_type;
    options.db_target_size = config.db_target_size;
    options.random_seed = config.random_seed;
    options.enable_stats = config.enable_stats;
    options.reinit = config.reinit;
    options.vector_file_path = config.vector_file_path;

    std::unique_ptr<lsm_vec::LSMVecDB> db;
    auto open_status = lsm_vec::LSMVecDB::Open(config.db_path, options, &db);
    if (!open_status.ok()) {
        std::cerr << "Failed to open LSMVecDB: " << open_status.ToString() << "\n";
        return 1;
    }

    std::cout << "Inserting nodes from " << config.input_file_path << std::endl;
    insertFromFile(*db, config.input_file_path);
    // std::vector<float> first_vec;
    // auto get_status = db->Get(0, &first_vec);
    // if (get_status.ok()) {
    //     std::cout << "Fetched vector for id 0 with " << first_vec.size() << " dims" << std::endl;
    // } else {
    //     std::cerr << "Get failed for id 0: " << get_status.ToString() << std::endl;
    // }

    std::cout << "Querying and comparing with ground truth " << config.query_file_path << std::endl;
    queryAndCompareWithGroundTruth(*db, config.query_file_path, config.groundtruth_file_path, config.k);
    if(options.enable_stats){
        std::cout << "---------------------------------" << std::endl;
        db->printStatistics();
        std::cout << "---------------------------------" << std::endl;
    }

    return 0;
    
    // This part is for testing close and reopen of LSMVecDB 
    db->Close();
    db.reset();
    options.reinit = false;
    open_status = lsm_vec::LSMVecDB::Open(config.db_path, options, &db);
    if (!open_status.ok()) {
        std::cerr << "Failed to open LSMVecDB: " << open_status.ToString() << "\n";
        return 1;
    }
    std::cout << "Querying and comparing with ground truth after reopen " << config.query_file_path << std::endl;
    queryAndCompareWithGroundTruth(*db, config.query_file_path, config.groundtruth_file_path, config.k);
    return 0;
}
