cmake_minimum_required (VERSION 3.3)
project (nbody_comp CXX)

list (APPEND CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -std=c++17")

set (ENV{KOKKOS_CXX_STANDARD} "c++17")
set (KOKKOS_PATH $ENV{KOKKOS_PATH} kokkos)
set (KOKKOS_ARCH "BDW")
set (KOKKOS_ENABLE_OPENMP)

add_subdirectory (${KOKKOS_PATH})
include_directories (${Kokkos_INCLUDE_DIRS_RET})

add_subdirectory (01_serial)
add_subdirectory (02_kokkos_1d_view)

add_subdirectory (04_kokkos_1d_view_tmp_storage)
add_subdirectory (05_kokkos_2d_view)
add_subdirectory (06_kokkos_2d_view_layout_left)
