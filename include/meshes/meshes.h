#ifndef SEM_MESHES_H
#define SEM_MESHES_H

namespace SEM { 
    /**
     * @brief Contains the different meshes used through the library.
     * 
     * These meshes can be used with the different solvers and polynomials, and have different properties.
     */
    namespace Meshes {}
}

#include "Mesh_t.cuh"
#include "Mesh_host_t.h"
#include "Mesh2D_t.cuh"

#endif