#ifndef SEM_SEM_H
#define SEM_SEM_H

/**
 * @brief Contains all the objects and functions of the library.
 * 
 * All the different namespaces are pulled under this one. 
 */
namespace SEM {
    /**
     * @brief Contains the host version of the objects and functions of the library.
     * 
     * All the different host namespaces are pulled under this one. 
     */
    namespace Host {}
}

#include "entities/entities.h"
#include "functions/functions.h"
#include "helpers/helpers.h"
#include "meshes/meshes.h"
#include "polynomials/polynomials.h"
#include "solvers/solvers.h"

#endif