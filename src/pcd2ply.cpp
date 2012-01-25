#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

int
main (int argc, char** argv)
{
  bool binary = false;

  if(argc < 3) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << argv[0] << "[-b] input.pcd output.ply" << std::endl;
    return (1);
  }

  if(argc == 4) {
    if(strncmp(argv[1],"-b",2) != 0) {
      std::cerr << "Error: unknown option!" << std::endl;
      return (1);
    }
    else {
      binary = true;
      argv += 1;
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PCDReader reader;
  reader.read<pcl::PointXYZ> (argv[1], *cloud);

  std::cerr << "Read cloud: " << std::endl;
  std::cerr << *cloud << std::endl;

  pcl::PLYWriter plywriter;
  plywriter.write<pcl::PointXYZ> (argv[2], *cloud, binary);

  return (0);
}