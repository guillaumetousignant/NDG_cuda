#ifndef NDG_ENTITIES_FACE2D_T_CUH
#define NDG_ENTITIES_FACE2D_T_CUH

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include "entities/cuda_vector.cuh"
#include <array>

namespace SEM { namespace Device { namespace Entities {

    /**
     * @brief The Face2D_t class describes an interface between two elements.
     * 
     * Faces have a polynomial order, which dictates the points on which element solution is projected. From the element
     * solution on the left and right of the face, the fluxes are computed. These fluxes can then be projected back to
     * the elements.
     */
    class Face2D_t {
        public:
            /**
             * @brief Constructs a new Face2D_t object from its polynomial order, two nodes, two neighbouring elements, and the two elements' sides connecting to the face.
             * 
             * @param N Polynomial order of the face.
             * @param nodes Array of the two nodes at the ends of the face. Numbering in the solution arrays goes from the first point to the second. [first, second]
             * @param elements Elements on the left and right of the face. The left element has nodes in the same order as the face, the right one is reversed. [left, right]
             * @param elements_side Side of the elements the face connects to. [left, right]
             */
            __device__ 
            Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements, std::array<size_t, 2> elements_side);

            /**
             * @brief Constructs an empty Face2D_t object.
             * 
             * This is the default initialisation constructor, everything is set to 0 and all vectors are empty.
             */
            __host__ __device__
            Face2D_t();

            int N_; /**< @brief Polynomial order of the face.*/

            // Connectivity
            std::array<size_t, 2> nodes_; /**< @brief Nodes making up the face. [first, second]*/
            std::array<size_t, 2> elements_; /**< @brief Elements connecting to the face. The left element has nodes in the same order as the face, the right one is reversed. [left, right]*/
            std::array<size_t, 2> elements_side_; /**< @brief Side of the elements the face connects to. [left, right]*/
            
            // Geometry
            SEM::Device::Entities::Vec2<deviceFloat> normal_; /**< @brief Normal vector of the face. Points from the first to the second element. Normalised.*/
            SEM::Device::Entities::Vec2<deviceFloat> tangent_; /**< @brief Tangent vector of the face. Points from the first to the second node. Normalised. */
            deviceFloat length_; /**< @brief Length of the face.*/
            std::array<deviceFloat, 2> offset_; /**< @brief Offset from the elements. 0 means the first node of the face coincides with the first node of the element side, 0.5 means it is in the middle, 1 means it coincides with the second node of the element side. [left, right]*/
            std::array<deviceFloat, 2> scale_; /**< @brief Scaling from the elements. 1 is the same length as the element, 0.5 is half, etc. [left, right]*/
            bool refine_; /**< @brief Face needs to be split in two.*/

            // Solution
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 2> p_; /**< @brief Pressure projected from both elements. [left, right], both sized N + 1.*/
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 2> u_; /**< @brief x velocity projected from both elements. [left, right], both sized N + 1.*/
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 2> v_; /**< @brief y velocity projected from both elements. [left, right], both sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> p_flux_; /**< @brief Pressure flux in the face. Sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> u_flux_; /**< @brief x velocity flux in the face. Sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> v_flux_; /**< @brief y velocity flux in the face. Sized N + 1.*/

            /**
             * @brief Creates the different vectors of the face according to its polynomial order, for when the face is copied from the host and the vectors are uninitialised.
             * 
             * When faces are created on the host and copied to the device the different solution vectors, like p_ and p_flux_, can't be initialised since they are not in shared memory.
             * This creates those vectors for the otherwise complete face.
             */
            __device__
            auto allocate_storage() -> void;

            /**
             * @brief Changes the polynomial order of the face and creates new vectors with the new size.
             */
            __device__
            auto resize_storage(int N) -> void;

            /**
             * @brief Sets the different vectors to null in case the face was not initialised.
             */
             __device__
             auto clear_storage() -> void;

            /**
             * @brief Computes the face's geometry from elements and nodes.
             *
             * Computes the face's normal, tangent, length, offset and scale, all from the first element to the second.
             * 
             * @param elements_centres Array of the center of each neighbour element.
             * @param nodes Array of nodes, in which the face's nodes are placed at their index.
             * @param element_nodes Array of arrays of the two nodes of each neighbour element.
             */
            __device__
            auto compute_geometry(const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2>& elements_centres, const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2>& nodes, const std::array<std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2>, 2>& element_nodes) -> void;
    };
}}}

#endif
