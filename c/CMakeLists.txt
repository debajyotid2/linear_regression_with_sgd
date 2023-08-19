cmake_minimum_required(VERSION 3.27.2)

project(linear_regression_with_sgd)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
add_executable(run main.c ${SOURCE_FILES})
target_include_directories(run PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(run PUBLIC m dl openblas Catch2Main Catch2)
