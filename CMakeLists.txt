cmake_minimum_required(VERSION 3.16)
project(cquantum C)

set(CMAKE_C_STANDARD 11)

# Add AVX-512 and other flags
if (MSVC)
    # AVX-512 support on MSVC
    add_compile_options(/O2 /arch:AVX512)
else()
    # GCC/Clang flags
    add_compile_options(-O2 -march=native -mavx512f -mavx512vl -fno-asynchronous-unwind-tables)
    add_compile_options(-fPIC)
endif()

# Source files
set(SOURCES
    main.c
    state_vector.c
    qgates.c
    vector_masks.c
    circuits.c
)

# Static library like `ar rcs cquantum.lib ...`
add_library(cquantum STATIC
    state_vector.c
    qgates.c
    vector_masks.c
    circuits.c
)

# Executable like `gcc ... -o avx512_m`
add_executable(avx512_m main.c)
target_link_libraries(avx512_m PRIVATE cquantum)

# Optional: Set output name of library
set_target_properties(cquantum PROPERTIES OUTPUT_NAME "cquantum")