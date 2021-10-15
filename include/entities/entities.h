#ifndef SEM_ENTITIES_ENTITIES_H
#define SEM_ENTITIES_ENTITIES_H

namespace SEM { namespace Host {
    /**
     * @brief Contains the basic entities used through the library.
     * 
     * Most of the entities represent interface classes, to be specialized in their respective namespaces.
     * Some entities, such as Vec2, represent basic types used throughout the program.
     */
    namespace Entities {}
}}

#include "Element_t.h"
#include "Element2D_t.h"
#include "Face_t.h"
#include "Face2D_t.h"
#include "NDG_t.h"
#include "Vec2.h"

#endif