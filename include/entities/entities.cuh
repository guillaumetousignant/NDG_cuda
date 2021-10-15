#ifndef SEM_ENTITIES_ENTITIES_CUH
#define SEM_ENTITIES_ENTITIES_CUH

namespace SEM { namespace Device {
    /**
     * @brief Contains the basic entities used through the library.
     * 
     * Most of the entities represent interface classes, to be specialized in their respective namespaces.
     * Some entities, such as Vec2, represent basic types used throughout the program.
     */
    namespace Entities {}
}}

#include "Element_t.cuh"
#include "Element2D_t.cuh"
#include "Face_t.cuh"
#include "Face2D_t.cuh"
#include "NDG_t.cuh"
#include "Vec2.cuh"
#include "cuda_vector.cuh"
#include "device_vector.cuh"
#include "host_vector.cuh"

#endif