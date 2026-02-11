#include "utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
// Function to read vectors from a bvecs file
std::vector<std::vector<float>> readBvecsFile(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<std::vector<float>> data;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (input)
    {
        int dim;
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!input)
            break;

        std::vector<uint8_t> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(uint8_t));

        if (input)
        {
            // Convert uint8_t to float for compatibility with HNSW
            std::vector<float> floatVec(vec.begin(), vec.end());
            data.push_back(floatVec);
        }
    }

    input.close();
    return data;
}

// Function to read ground truth from an ivecs file
std::vector<std::vector<int>> readIvecsFile(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<std::vector<int>> data;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (input)
    {
        int k;
        input.read(reinterpret_cast<char *>(&k), sizeof(int));
        if (!input)
            break;

        std::vector<int> vec(k);
        input.read(reinterpret_cast<char *>(vec.data()), k * sizeof(int));
        if (input)
        {
            data.push_back(vec);
        }
    }

    input.close();
    return data;
}

// Function to insert vectors from a bvecs file into the HNSW graph
void insertFromBigANNFile(lsm_vec::LSMVecDB &db, const std::string &filename)
{
    auto data = readBvecsFile(filename);
    for (size_t i = 0; i < data.size(); ++i)
    {
        auto status = db.Insert(static_cast<lsm_vec::node_id_t>(i), lsm_vec::Span<float>(data[i]));
        if (!status.ok())
        {
            std::cerr << "Insert failed for node " << i << ": " << status.ToString() << std::endl;
            return;
        }
    }
}

// Function to perform queries from a file and compare results with ground truth
void queryBigANN(lsm_vec::LSMVecDB &db, const std::string &queryFile, const std::string &groundTruthFile)
{
    auto queries = readBvecsFile(queryFile);
    auto groundTruth = readIvecsFile(groundTruthFile);

    if (queries.size() != groundTruth.size())
    {
        std::cerr << "Error: The number of queries and ground truth entries do not match." << std::endl;
        return;
    }

    int correctMatches = 0;
    int totalQueries = static_cast<int>(queries.size());
    lsm_vec::SearchOptions search_options;
    search_options.k = 1;

    for (size_t i = 0; i < queries.size(); ++i)
    {
        std::vector<lsm_vec::SearchResult> results;
        auto status = db.SearchKnn(lsm_vec::Span<float>(queries[i]), search_options, &results);
        if (!status.ok() || results.empty())
        {
            std::cerr << "Search failed for query " << i << ": " << status.ToString() << std::endl;
            continue;
        }
        int hnswResult = results.front().id;
        int groundTruthResult = groundTruth[i][0]; // Assuming the first entry is the closest

        if (hnswResult == groundTruthResult)
        {
            ++correctMatches;
        }

        std::cout << "Query " << i << ": HNSW Nearest Neighbor = " << hnswResult
                  << ", Ground Truth = " << groundTruthResult << std::endl;
    }

    // Calculate and print accuracy
    float accuracy = static_cast<float>(correctMatches) / totalQueries;
    std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;
}

std::vector<std::vector<float>> readFvecsFile(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<std::vector<float>> data;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (input)
    {
        int dim;
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!input)
            break;

        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (input)
        {
            data.push_back(vec);
        }
    }

    input.close();
    return data;
}

int getdim(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    int dim = 0;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return dim;
    }

    while (input)
    {
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        break;
    }

    input.close();
    return dim;
}

// Function to insert vectors from an fvecs file into the HNSW graph
void insertFromFile(lsm_vec::LSMVecDB &db, const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    int dim = 0;
    size_t node_count = 0;

    auto start = std::chrono::high_resolution_clock::now();

    bool isFvecs = filename.find(".fvecs") != std::string::npos;
    bool isBvecs = filename.find(".bvecs") != std::string::npos;

    if (!isFvecs && !isBvecs)
    {
        std::cerr << "Error: Unsupported file format. Only fvecs and bvecs files are supported." << std::endl;
        return;
    }

    while (input.read(reinterpret_cast<char *>(&dim), sizeof(int)))
    {
        if (node_count == 0)
        {
            std::cout << "vector dim: " << dim << std::endl;
        }

        std::vector<float> floatVec;
        std::vector<float> finalVec(dim);

        if (isFvecs)
        {
            floatVec.resize(dim);
            input.read(reinterpret_cast<char *>(floatVec.data()), dim * sizeof(float));
            finalVec = floatVec;
        }
        else if (isBvecs)
        {
            std::vector<uint8_t> byteVec(dim);
            input.read(reinterpret_cast<char *>(byteVec.data()), dim * sizeof(uint8_t));

            
            for (int j = 0; j < dim; ++j)
            {
                finalVec[j] = static_cast<float>(byteVec[j]);
            }
        }

        auto status = db.Insert(static_cast<lsm_vec::node_id_t>(node_count), lsm_vec::Span<float>(finalVec));
        if (!status.ok())
        {
            std::cerr << "Insert failed for node " << node_count << ": " << status.ToString() << std::endl;
            break;
        }

        if (node_count % 1000 == 0)
        {
            std::cout << "Inserting node " << node_count << std::endl;
        }

        ++node_count;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Building " << node_count << " nodes took " << duration << " seconds" << std::endl;

    input.close();
}

// Function to perform queries from a file and compare results with ground truth
void queryAndCompareWithGroundTruth(lsm_vec::LSMVecDB &db,
                                    const std::string &queryFile,
                                    const std::string &groundTruthFile,
                                    int k)
{
    auto queries = readFvecsFile(queryFile);
    auto groundTruth = readIvecsFile(groundTruthFile);

    if (queries.size() != groundTruth.size())
    {
        std::cerr << "Error: The number of queries and ground truth entries do not match." << std::endl;
        return;
    }

    int correctMatches = 0;
    int totalQueries = static_cast<int>(queries.size());
    double totalQueryTime = 0.0;
    lsm_vec::SearchOptions search_options;
    search_options.k = k;

    for (size_t i = 0; i < queries.size(); ++i)
    {
        // Start measuring query time
        auto start = std::chrono::high_resolution_clock::now();

        // Perform the HNSW query
        std::vector<lsm_vec::SearchResult> results;
        auto status = db.SearchKnn(lsm_vec::Span<float>(queries[i]), search_options, &results);

        // Stop measuring query time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> queryTime = end - start;

        // Accumulate query time
        totalQueryTime += queryTime.count();

        if (!status.ok() || results.empty())
        {
            std::cerr << "Search failed for query " << i << ": " << status.ToString() << std::endl;
            continue;
        }

        int max_truth = std::min(static_cast<int>(groundTruth[i].size()), k);
        bool matched = false;
        for (const auto& result : results) {
            for (int idx = 0; idx < max_truth; ++idx) {
                if (result.id == groundTruth[i][idx]) {
                    matched = true;
                    break;
                }
            }
            if (matched) {
                break;
            }
        }
        if (matched) {
            ++correctMatches;
        }
    }

    // Calculate and print accuracy
    float accuracy = static_cast<float>(correctMatches) / totalQueries;
    std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;

    // Print efficiency statistics
    std::cout << "Total Query Time: " << totalQueryTime << " ms" << std::endl;
    std::cout << "Average Query Time: " << (totalQueryTime / totalQueries) << " ms/query" << std::endl;
}
