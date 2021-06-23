#ifndef NDG_FACE2D_T_H
#define NDG_FACE2D_T_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include "entities/cuda_vector.cuh"
#include <array>

namespace SEM { namespace Entities {
    class Face2D_t {
        public:
            __device__ 
            Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements, std::array<size_t, 2> elements_size);

            __host__ __device__
            Face2D_t();

            int N_; /**< @brief Polynomial order of the face.*/

            // Connectivity
            std::array<size_t, 2> nodes_; /**< @brief Nodes making up the face. [left, right]*/
            std::array<size_t, 2> elements_; /**< @brief Elements connecting to the face. [left, right]*/
            std::array<size_t, 2> elements_side_; /**< @brief Side of the elements the face connects to. [left, right]*/
            std::array<deviceFloat, 2> offset_; /**< @brief Offset from the elements. [left, right]*/
            std::array<deviceFloat, 2> scale_; /**< @brief Scaling from the elements. [left, right]*/

            // Geometry
            SEM::Entities::Vec2<deviceFloat> normal_; /**< @brief Normal vector of the face. Points from the first to the second element. Normalised.*/
            SEM::Entities::Vec2<deviceFloat> tangent_; /**< @brief Tangent vector of the face. Points from the first to the second node. Normalised. */
            deviceFloat length_; /**< @brief Length of the face.*/

            // Solution
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 2> p_; /**< @brief Pressure in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 2> u_; /**< @brief x velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 2> v_; /**< @brief y velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Entities::cuda_vector<deviceFloat> p_flux_; /**< @brief Pressure flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Entities::cuda_vector<deviceFloat> u_flux_; /**< @brief x velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Entities::cuda_vector<deviceFloat> v_flux_; /**< @brief y velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/

            __device__
            auto allocate_storage() -> void;
    };
}}

#endif
