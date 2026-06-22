#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <bitset>
#include <unordered_map>
#include <iostream>

namespace astervec
{
    /**
     * SimHash-based Sampling Module for LSM-VEC
     * Implements the sampling strategy described in Section III-C of the paper
     */
    class SimHashSampler
    {
    private:
        int hashBits;                           // Number of hash bits (m in paper)
        int vectorDim;                          // Dimension of vectors
        std::vector<std::vector<float>> projectionVectors; // Random projection vectors
        
        // Hash codes for all vectors (stored in memory)
        std::unordered_map<int, std::vector<int8_t>> hashCodes;
        
        // Statistics
        mutable size_t totalComparisons;
        mutable size_t filteredComparisons;
        
        std::mt19937 rng;

    public:
        /**
         * Constructor
         * @param hashBits Number of projection vectors (m in paper, default 32)
         * @param vectorDim Dimension of data vectors
         */
        SimHashSampler(int hashBits = 32, int vectorDim = 128)
            : hashBits(hashBits), vectorDim(vectorDim), 
              totalComparisons(0), filteredComparisons(0)
        {
            rng.seed(12345);
            initializeProjectionVectors();
        }

        /**
         * Initialize random projection vectors
         * Each projection vector is sampled from N(0, I_d)
         */
        void initializeProjectionVectors()
        {
            projectionVectors.resize(hashBits);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            for (int i = 0; i < hashBits; ++i)
            {
                projectionVectors[i].resize(vectorDim);
                for (int j = 0; j < vectorDim; ++j)
                {
                    projectionVectors[i][j] = dist(rng);
                }
                
                // Normalize projection vector
                float norm = 0.0f;
                for (int j = 0; j < vectorDim; ++j)
                {
                    norm += projectionVectors[i][j] * projectionVectors[i][j];
                }
                norm = std::sqrt(norm);
                
                for (int j = 0; j < vectorDim; ++j)
                {
                    projectionVectors[i][j] /= norm;
                }
            }
            
            std::cout << "Initialized " << hashBits << " projection vectors for "
                      << vectorDim << "-dimensional space" << std::endl;
        }

        /**
         * Compute SimHash code for a vector
         * Hash(x) = [sgn(x^T * a_1), sgn(x^T * a_2), ..., sgn(x^T * a_m)]
         * 
         * @param vector Input vector
         * @return Hash code as vector of {-1, 1}
         */
        std::vector<int8_t> computeHash(const std::vector<float> &vector)
        {
            std::vector<int8_t> hashCode(hashBits);
            
            for (int i = 0; i < hashBits; ++i)
            {
                float dotProduct = 0.0f;
                for (int j = 0; j < vectorDim; ++j)
                {
                    dotProduct += vector[j] * projectionVectors[i][j];
                }
                
                // sgn function: 1 if >= 0, -1 otherwise
                hashCode[i] = (dotProduct >= 0.0f) ? 1 : -1;
            }
            
            return hashCode;
        }

        /**
         * Store hash code for a vector ID
         * Called during vector insertion
         */
        void storeHash(int vectorId, const std::vector<float> &vector)
        {
            hashCodes[vectorId] = computeHash(vector);
        }

        /**
         * Count hash collisions between two vectors
         * #Col(q, u) = 0.5 * (m + Hash(q)^T · Hash(u))
         * 
         * @param hash1 First hash code
         * @param hash2 Second hash code
         * @return Number of matching bits
         */
        int countCollisions(const std::vector<int8_t> &hash1, 
                           const std::vector<int8_t> &hash2) const
        {
            int collisions = 0;
            for (int i = 0; i < hashBits; ++i)
            {
                collisions += (hash1[i] == hash2[i]) ? 1 : 0;
            }
            return collisions;
        }

        /**
         * Compute collision threshold based on Hoeffding's inequality
         * T_SimHash_ε = threshold number of collisions
         * 
         * @param epsilon Error probability (default 0.1)
         * @param delta Maximum distance to current k-th nearest neighbor
         * @return Collision threshold
         */
        int computeCollisionThreshold(float epsilon = 0.1f, float delta = 1.0f) const
        {
            // Simplified threshold computation
            // In practice, this should be calibrated based on the dataset
            // Using formula from paper: based on angle probability
            
            // θ = arccos(1 - δ²/2) approximation for Euclidean distance
            // P(collision) ≈ 1 - θ/π
            
            float theta = std::acos(1.0f - delta * delta / 2.0f);
            float collisionProb = 1.0f - theta / M_PI;
            
            // Hoeffding bound: threshold = m * p - sqrt(m * log(1/ε) / 2)
            float threshold = hashBits * collisionProb - 
                            std::sqrt(hashBits * std::log(1.0f / epsilon) / 2.0f);
            
            return std::max(1, static_cast<int>(threshold));
        }


        bool shouldKeepCandidate(const std::vector<int8_t> &queryHash, 
                                int candidateId, 
                                int threshold)
        {
            totalComparisons++;
            
            // If hash code not found, keep the candidate (conservative)
            if (hashCodes.find(candidateId) == hashCodes.end())
            {
                return true;
            }
            
            int collisions = countCollisions(queryHash, hashCodes[candidateId]);
            
            bool keep = (collisions >= threshold);
            if (!keep)
            {
                filteredComparisons++;
            }
            
            return keep;
        }

        std::vector<int> sampleNeighbors(const std::vector<int8_t> &queryHash,
                                         const std::vector<int> &neighbors,
                                         float samplingRatio,
                                         int threshold)
        {
            std::vector<int> sampled;
            
            if (samplingRatio >= 1.0f)
            {
                // No sampling, return all neighbors that pass filter
                for (int neighbor : neighbors)
                {
                    if (shouldKeepCandidate(queryHash, neighbor, threshold))
                    {
                        sampled.push_back(neighbor);
                    }
                }
            }
            else
            {
                // Sample with collision-based filtering
                for (int neighbor : neighbors)
                {
                    // First apply hash filter
                    if (shouldKeepCandidate(queryHash, neighbor, threshold))
                    {
                        // Then apply random sampling
                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                        if (dist(rng) < samplingRatio)
                        {
                            sampled.push_back(neighbor);
                        }
                    }
                }
            }
            
            return sampled;
        }

        /**
         * Get hash code for a vector ID
         */
        const std::vector<int8_t>* getHashCode(int vectorId) const
        {
            auto it = hashCodes.find(vectorId);
            if (it != hashCodes.end())
            {
                return &(it->second);
            }
            return nullptr;
        }

        /**
         * Remove hash code for deleted vector
         */
        void removeHash(int vectorId)
        {
            hashCodes.erase(vectorId);
        }

        /**
         * Print statistics
         */
        void printStatistics() const
        {
            std::cout << "=== SimHash Sampling Statistics ===" << std::endl;
            std::cout << "Total comparisons: " << totalComparisons << std::endl;
            std::cout << "Filtered comparisons: " << filteredComparisons << std::endl;
            if (totalComparisons > 0)
            {
                float filterRate = 100.0f * filteredComparisons / totalComparisons;
                std::cout << "Filter rate: " << filterRate << "%" << std::endl;
            }
            std::cout << "Hash codes stored: " << hashCodes.size() << std::endl;
        }

        /**
         * Reset statistics
         */
        void resetStatistics()
        {
            totalComparisons = 0;
            filteredComparisons = 0;
        }

        /**
         * Get memory usage
         */
        size_t getMemoryUsage() const
        {
            // Projection vectors: hashBits * vectorDim * sizeof(float)
            size_t projMemory = hashBits * vectorDim * sizeof(float);
            
            // Hash codes: hashCodes.size() * hashBits * sizeof(int8_t)
            size_t hashMemory = hashCodes.size() * hashBits * sizeof(int8_t);
            
            return projMemory + hashMemory;
        }
    };

} // namespace ROCKSDB_NAMESPACE
