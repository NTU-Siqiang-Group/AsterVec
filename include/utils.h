#ifndef HNSW_UTILS_H
#define HNSW_UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include "lsm_vec_db.h"

// Reads vectors from a bvecs file
std::vector<std::vector<float>> readBvecsFile(const std::string& filename);

// Reads ground truth indices from an ivecs file
std::vector<std::vector<int>> readIvecsFile(const std::string& filename);

// Inserts vectors from a bvecs file into the HNSW graph
void insertFromBigANNFile(lsm_vec::LSMVecDB& db, const std::string& filename);

// Queries the HNSW graph and compares results with ground truth
void queryBigANN(lsm_vec::LSMVecDB& db, const std::string& queryFile, const std::string& groundTruthFile);

std::vector<std::vector<float>> readFvecsFile(const std::string& filename);

void insertFromFile(lsm_vec::LSMVecDB& db, const std::string& filename);

void queryAndCompareWithGroundTruth(lsm_vec::LSMVecDB& db,
                                    const std::string& queryFile,
                                    const std::string& groundTruthFile,
                                    int k,
                                    int ef_search = 64);

int getdim(const std::string& filename);
#endif // HNSW_UTILS_H
