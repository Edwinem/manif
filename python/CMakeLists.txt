add_subdirectory(pybind11)



include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/lt)


set(PYBIND11_CPP_STANDARD -std=c++11)
pybind11_add_module(PyManif manif_wrapper/manif_bindings.cpp)