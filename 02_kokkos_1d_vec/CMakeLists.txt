# Setting up Kokkos for everything
add_subdirectory (${KOKKOS_PATH})
include_directories (${Kokkos_INCLUDE_DIRS_RET})

add_executable (02_kokkos_1d_vec main.cpp)
target_link_libraries (02_kokkos_1d_vec kokkos)

