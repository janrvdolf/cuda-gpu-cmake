cmake_minimum_required(VERSION 3.16)
project(untitled1 CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CUDACXX /usr/local/cuda-11.0/bin/nvcc)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)

add_executable(untitled1 main.cu)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})


set_target_properties(
        untitled1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# reverseArray_reseni.cu
add_executable(reseni reverseArray_reseni.cu)

set_target_properties(
        reseni
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(life_game life_game_template.cu)

set_target_properties(
        life_game
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(life_game ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
