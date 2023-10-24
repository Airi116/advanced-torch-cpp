cmake_minimum_required(VERSION 3.10)

project(googletest-download NONE)

include(ExternalProject)

ExternalProject_Add(
  googletest
  SOURCE_DIR "@GO