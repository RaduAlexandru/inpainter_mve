# compile your programm using the following commands:
#
# Enter build folder
# $ cd build
#
# !!! To Make an Eclipse Project you should use the Generator in the following way
# .../inpainter/build/$ cmake -G"Eclipse CDT4 - Unix Makefiles" ../
# .../inpainter/build/$ make

# OR

# !!! To Make an ordinary Unix Makefile project you can use the Generator in the following way
# .../inpainter/build/$ cmake -G"Unix Makefiles" ../
# .../inpainter/build/$ make


cmake_minimum_required(VERSION 2.6)
project (inpainter)




file(GLOB inpainter_SRC
    src/*.cpp
    src/*.h
    src/*.cc
)



find_package(Eigen 3 QUIET)
if(NOT Eigen_FOUND)
  include(${CMAKE_SOURCE_DIR}/cfg/FindEigen3.cmake)
  set(Eigen_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
endif()
include_directories(${Eigen_INCLUDE_DIR})



find_package(Boost COMPONENTS thread date_time program_options filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})








SET(CMAKE_CXX_FLAGS "-std=c++0x")				# New C11
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)			# Binary file location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})			# Exectuable file location for main and subprojects
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)		# Library file location for main and subprojects
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)			# Source folder
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)		# Header folder

#set(CMAKE_BUILD_TYPE debug)					# choose if you want to compile with or without debugging flags,
set(CMAKE_BUILD_TYPE release)					# uncomment only one of the following two commands at a time

#To build the project openCV is required
find_package( OpenCV REQUIRED )

#Include Required Directories
include_directories(${PROJECT_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})



# Create a shared library (To use library creation uncomment the following)
#	add_library(
#		LIBRARY_NAME
#		${PROJECT_SOURCE_DIR}/some.cpp
#		${PROJECT_SOURCE_DIR}/someMore.cpp
#		)

#Create an Executable for segment
add_executable(inpainter ${PROJECT_SOURCE_DIR}/inpainter.cpp  ${inpainter_SRC} )	#Add executable to inpainter.cpp
target_link_libraries( inpainter ${OpenCV_LIBS} ${Boost_LIBRARIES})			#Link the executable with opencv library

#(If you want to add subprojects to your main the following is how you do)
#Continue with subdirectory
# add_subdirectory(SubFolder/That/Contains/Another/CMakeLists.txt)
