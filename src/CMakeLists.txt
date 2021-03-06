
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(fenzi)

SET(fenzi_VERSION_MAJOR 0)
SET(fenzi_VERSION_MINOR 1)

FIND_PACKAGE(CUDA)

IF(CUDA_VERSION VERSION_LESS "3.0")
    SET(CUDA_HOST_COMPILATION_CPP OFF)
ENDIF()

SET(CUDA_ARCHITECTURE sm_13
    CACHE STRING "GPU Architecture (sm_13, sm_20)")

SET(FENZI_SYSTEM_SIZE SMALL
    CACHE STRING
    "FENZI system size (VSMALL, SMALL, MEDIUM, LARGE, CCO, CUSTOM)")


SET(FENZI_PCONST_DOCSTRING "run FEN ZI in npt ensamble")
SET(FENZI_NPT_DOCSTRING "run FEN ZI in npt ensamble (with shake)")
SET(FENZI_CONSFIX_DOCSTRING "run FEN ZI with consfix")

SET(FENZI_PCONST  0 CACHE BOOL ${FENZI_PCONST_DOCSTRING})
SET(FENZI_NPT     0 CACHE BOOL ${FENZI_NPT_DOCSTRING})
SET(FENZI_CONSFIX 0 CACHE BOOL ${FENZI_CONSFIX_DOCSTRING})

SET(temp 0)
IF(FENZI_PCONST)
    SET(temp 1)
ENDIF()
SET(FENZI_PCONST ${temp} CACHE BOOL ${FENZI_PCONST_DOCSTRING} FORCE)

SET(temp 0)
IF(FENZI_NPT)
    SET(temp 1)
    SET(FENZI_PCONST ${temp} CACHE BOOL ${FENZI_PCONST_DOCSTRING} FORCE)
ENDIF()
SET(FENZI_NPT ${temp} CACHE BOOL ${FENZI_NPT_DOCSTRING} FORCE)

SET(temp 0)
IF(FENZI_CONSFIX)
    SET(temp 1)
ENDIF()
SET(FENZI_CONSFIX ${temp} CACHE BOOL ${FENZI_CONSFIX_DOCSTRING} FORCE)

#  .-------------------------------------------------.
#  |               PARAMETER TABLE                   |
#  |-------------------------------------------------|
#  |                 VSMALL SMALL MEDIUM LARGE   CCO |
#  | CELL_BLOCKSIZE      96   128    128   256   256 |
#  | CELL_ATOMS          96   128    128   256   256 |
#  |                                                 |
#  | MAX_DIHED_TYPE      64    64     64    64   484 |
#  | MAX_ANGLE_TYPE      64    64     64    64   318 |
#  | MAX_BOND_TYPE       64    64     64    64   256 |
#  '-------------------------------------------------'

SET(FENZI_CELL_BLOCKSIZE_DOCSTRING "Cell Blocksize"          )
SET(FENZI_CELL_ATOMS_DOCSTRING     "Max # of atoms in a cell")
SET(FENZI_MAX_DIHED_TYPE_DOCSTRING "Max # of dihedral types" )
SET(FENZI_MAX_ANGLE_TYPE_DOCSTRING "Max # of angle types"    )
SET(FENZI_MAX_BOND_TYPE_DOCSTRING  "Max # of bond types"     )

IF(FENZI_SYSTEM_SIZE STREQUAL "VSMALL")
    SET(FENZI_CELL_BLOCKSIZE  96
        CACHE STRING ${FENZI_CELL_BLOCKSIZE_DOCSTRING} FORCE)
    SET(FENZI_CELL_ATOMS      96
        CACHE STRING ${FENZI_CELL_ATOMS_DOCSTRING}     FORCE)
    SET(FENZI_MAX_DIHED_TYPE  64
        CACHE STRING ${FENZI_MAX_DIHED_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_ANGLE_TYPE  64
        CACHE STRING ${FENZI_MAX_ANGLE_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_BOND_TYPE   64
        CACHE STRING ${FENZI_MAX_BOND_TYPE_DOCSTRING}  FORCE)
ELSEIF(FENZI_SYSTEM_SIZE STREQUAL "SMALL")
    SET(FENZI_CELL_BLOCKSIZE 128
        CACHE STRING ${FENZI_CELL_BLOCKSIZE_DOCSTRING} FORCE)
    SET(FENZI_CELL_ATOMS     128
        CACHE STRING ${FENZI_CELL_ATOMS_DOCSTRING}     FORCE)
    SET(FENZI_MAX_DIHED_TYPE  64
        CACHE STRING ${FENZI_MAX_DIHED_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_ANGLE_TYPE  64
        CACHE STRING ${FENZI_MAX_ANGLE_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_BOND_TYPE   64
        CACHE STRING ${FENZI_MAX_BOND_TYPE_DOCSTRING}  FORCE)
ELSEIF(FENZI_SYSTEM_SIZE STREQUAL "MEDIUM")
    SET(FENZI_CELL_BLOCKSIZE 128
        CACHE STRING ${FENZI_CELL_BLOCKSIZE_DOCSTRING} FORCE)
    SET(FENZI_CELL_ATOMS     128
        CACHE STRING ${FENZI_CELL_ATOMS_DOCSTRING}     FORCE)
    SET(FENZI_MAX_DIHED_TYPE  64
        CACHE STRING ${FENZI_MAX_DIHED_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_ANGLE_TYPE  64
        CACHE STRING ${FENZI_MAX_ANGLE_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_BOND_TYPE   64
        CACHE STRING ${FENZI_MAX_BOND_TYPE_DOCSTRING}  FORCE)
ELSEIF(FENZI_SYSTEM_SIZE STREQUAL "LARGE")
    SET(FENZI_CELL_BLOCKSIZE 256
        CACHE STRING ${FENZI_CELL_BLOCKSIZE_DOCSTRING} FORCE)
    SET(FENZI_CELL_ATOMS     256
        CACHE STRING ${FENZI_CELL_ATOMS_DOCSTRING}     FORCE)
    SET(FENZI_MAX_DIHED_TYPE  64
        CACHE STRING ${FENZI_MAX_DIHED_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_ANGLE_TYPE  64
        CACHE STRING ${FENZI_MAX_ANGLE_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_BOND_TYPE   64
        CACHE STRING ${FENZI_MAX_BOND_TYPE_DOCSTRING}  FORCE)
ELSEIF(FENZI_SYSTEM_SIZE STREQUAL "CCO")
    SET(FENZI_CELL_BLOCKSIZE 256
        CACHE STRING ${FENZI_CELL_BLOCKSIZE_DOCSTRING} FORCE)
    SET(FENZI_CELL_ATOMS     256
        CACHE STRING ${FENZI_CELL_ATOMS_DOCSTRING}     FORCE)
    SET(FENZI_MAX_DIHED_TYPE 484
        CACHE STRING ${FENZI_MAX_DIHED_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_ANGLE_TYPE 318
        CACHE STRING ${FENZI_MAX_ANGLE_TYPE_DOCSTRING} FORCE)
    SET(FENZI_MAX_BOND_TYPE  256
        CACHE STRING ${FENZI_MAX_BOND_TYPE_DOCSTRING}  FORCE)
ENDIF()

SET(MAXNB 580
    CACHE STRING "Maximum number of non-bonded neighbors")

SET(MAXLATTICE_NEIGHBORNS 64
    CACHE STRING "Maximum number of non-bonded neighbors")

MARK_AS_ADVANCED(FENZI_CELL_BLOCKSIZE FENZI_CELL_ATOMS
                 FENZI_MAX_DIHED_TYPE FENZI_MAX_ANGLE_TYPE
                 FENZI_MAX_BOND_TYPE                      )

SET(SYSTEM_SIZE "SYSTEM_SIZE_${FENZI_SYSTEM_SIZE}")

CONFIGURE_FILE ( "${PROJECT_SOURCE_DIR}/defs.in.h"
                 "${PROJECT_BINARY_DIR}/defs.h"   )

# add the binary tree to the search path for include files
# so that we will find the generated header file
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")
CUDA_INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch ${CUDA_ARCHITECTURE})

# -O3 -use_fast_math --ptxas-options="-v -O3"

SET(target fenzi_${fenzi_VERSION_MAJOR}_${fenzi_VERSION_MINOR})
CUDA_ADD_EXECUTABLE(${target} fenzi.cu)
CUDA_ADD_CUFFT_TO_TARGET(${target})

