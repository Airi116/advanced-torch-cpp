cmake_minimum_required(VERSION 3.10)

if (NOT TARGET gtest)
  include(googletest)
  __fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}_googletest
)
endif()

set(DATA_PATH "${PROJECT_SOURCE_DIR}/data")
configure_file (conf