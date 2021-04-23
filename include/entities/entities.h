#ifndef SEM_ENTITIES_H
#define SEM_ENTITIES_H

namespace SEM { 
    /**
     * @brief Contains the basic entities used through the library.
     * 
     * Most of the entities represent interface classes, to be specialized in their respective namespaces.
     * Some entities, such as Vec2, represent basic types used throughout the program.
     */
    namespace Entities {}
}

#include "Element_t.cuh"
#include "Element2D_t.cuh"
#include "Element_host_t.h"
#include "Face_t.cuh"
#include "Face2D_t.cuh"
#include "Face_host_t.h"
#include "NDG_t.cuh"
#include "NDG_host_t.h"
#include "Vec2.cuh"
#include "device_vector.cuh"

#endif