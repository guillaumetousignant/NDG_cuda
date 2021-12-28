#ifndef SEM_SEM_CUH
#define SEM_SEM_CUH

/**
 * @brief Contains all the objects and functions of the library.
 * 
 * All the different namespaces are pulled under this one. 
 */
namespace SEM {
    /**
     * @brief Contains the device version of the objects and functions of the library.
     * 
     * All the different device namespaces are pulled under this one. 
     */
    namespace Device {}
}

#include "entities/entities.cuh"
#include "functions/functions.cuh"
#include "helpers/helpers.cuh"
#include "meshes/meshes.cuh"
#include "polynomials/polynomials.cuh"
#include "solvers/solvers.cuh"

#endif