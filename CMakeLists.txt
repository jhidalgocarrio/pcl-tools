cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcl-tools)

find_package(PCL 1.7 REQUIRED)
find_package(VTK REQUIRED)
add_definitions(-std=c++11)
find_package(Boost COMPONENTS system REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${VTK_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (statistical_removal src/statistical_removal.cpp)
target_link_libraries (statistical_removal ${PCL_LIBRARIES})
add_executable (pcd2ply src/pcd2ply.cpp)
target_link_libraries (pcd2ply ${PCL_LIBRARIES})
add_executable (pcdnormal2ply src/pcdnormal2ply.cpp)
target_link_libraries (pcdnormal2ply ${PCL_LIBRARIES})
add_executable (ply2pcd src/ply2pcd.cpp)
target_link_libraries (ply2pcd ${PCL_LIBRARIES})
add_executable (normal_estimation_omp src/normal_estimation_omp.cpp)
target_link_libraries (normal_estimation_omp ${PCL_LIBRARIES})
add_executable(pcd_viewer src/pcd_viewer.cpp)
target_link_libraries(pcd_viewer ${VTK_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES})

add_executable(features_demo src/features_demo.cpp)
target_link_libraries(features_demo ${PCL_LIBRARIES})
add_executable (correspondence_grouping src/correspondence_grouping.cpp)
target_link_libraries (correspondence_grouping ${PCL_LIBRARIES})
add_executable (narf_feature_extraction src/narf_feature_extraction.cpp)
target_link_libraries (narf_feature_extraction ${PCL_LIBRARIES})


IF(APPLE)
  add_executable(PCDViewer MACOSX_BUNDLE src/pcd_viewer.cpp)
  target_link_libraries(PCDViewer ${VTK_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES})
ENDIF(APPLE)
