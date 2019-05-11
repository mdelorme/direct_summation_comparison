cmake_minimum_required (VERSION 3.3)
project (nbody_comp CXX)

list (APPEND CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -O3)

set (KOKKOS_PATH $ENV{KOKKOS_PATH} kokkos)
set (KOKKOS_ARCH "BDW")

add_subdirectory (01_serial)
add_subdirectory (02_kokkos_1d_vec)
