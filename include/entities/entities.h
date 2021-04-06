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

#include "Vec2.cuh"
#include "device_vector.cuh"

#endif