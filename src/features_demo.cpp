#include <iostream>   // std::cout
#include <string>     // std::string, std::stof
#include <vector>
#include <Eigen/Core>
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include "pcl/io/pcd_io.h"
#include "pcl/io/ply_io.h"
#include "pcl/kdtree/kdtree_flann.h"
#include <pcl/range_image/range_image.h>
#include "pcl/features/normal_3d.h"
#include "pcl/features/pfh.h"
#include <pcl/features/fpfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/range_image_border_extractor.h>
#include "pcl/keypoints/sift_keypoint.h"
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/conversions.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>


typedef pcl::PCLPointCloud2 PCLPointCloud2;
typedef typename PCLPointCloud2::Ptr PCLPointCloud2Ptr;

void
downsample (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points, float leaf_size,
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &downsampled_out)
{

  pcl::VoxelGrid<pcl::PointXYZRGBA> vox_grid;
  vox_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
  vox_grid.setInputCloud (points);
  vox_grid.filter (*downsampled_out);
}

void
compute_surface_normals (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points, float normal_radius,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
{
  pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> norm_est;

  // Use a FLANN-based KdTree to perform neighborhood searches
  //norm_est.setSearchMethod (pcl::KdTreeFLANN<pcl::PointXYZRGBA>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZRGBA>));
  norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGBA>));

  // Specify the size of the local neighborhood to use when computing the surface normals
  norm_est.setRadiusSearch (normal_radius);

  // Set the input points
  norm_est.setInputCloud (points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute (*normals_out);
}

void
compute_PFH_features (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points,
                      pcl::PointCloud<pcl::Normal>::Ptr &normals,
                      float feature_radius,
                      pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr &descriptors_out)
{
  // Create a PFHRGBEstimation object
  pcl::PFHRGBEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::PFHRGBSignature250> pfh_est;

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  pfh_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGBA>));

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch (feature_radius);

  // Set the input points and surface normals
  pfh_est.setInputCloud (points);
  pfh_est.setInputNormals (normals);

  // Compute the features
  pfh_est.compute (*descriptors_out);
}

void
compute_SHOT_features (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points,
                      pcl::PointCloud<pcl::Normal>::Ptr &normals,
                      float feature_radius,
                      pcl::PointCloud<pcl::SHOT352>::Ptr &descriptors_out)
{
    // Create a SHOTEstimation object
    pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> descr_est;

    descr_est.setInputCloud (points);
    descr_est.setInputNormals (normals);
    descr_est.setSearchSurface (points);
    descr_est.compute (*descriptors_out);
}


void
detect_keypoints (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points,
                  float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast,
                  pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out)
{
  pcl::SIFTKeypoint<pcl::PointXYZRGBA, pcl::PointWithScale> sift_detect;

  // Use a FLANN-based KdTree to perform neighborhood searches
  sift_detect.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGBA>));

  // Set the detection parameters
  sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast (min_contrast);

  // Set the input
  sift_detect.setInputCloud (points);

  // Detect the keypoints and store them in "keypoints_out"
  sift_detect.compute (*keypoints_out);
}

void
detect_narf_keypoints (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points,
                    float angular_resolution, float noise_level, float min_range, int border_size,
                    float support_size,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &keypoints_out)
{
    Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;

    /** Compute range image **/
    boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);
    pcl::RangeImage& range_image = *range_image_ptr;
    range_image.createFromPointCloud (*points, angular_resolution,
                                    pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                    scene_sensor_pose, coordinate_frame,
                                    noise_level, min_range, border_size);

    /** Extract NARF keypoints **/
    pcl::RangeImageBorderExtractor range_image_border_extractor;
    pcl::NarfKeypoint narf_keypoint_detector;
    narf_keypoint_detector.setRangeImageBorderExtractor (&range_image_border_extractor);
    narf_keypoint_detector.setRangeImage (&range_image);
    narf_keypoint_detector.getParameters().support_size = support_size;

    pcl::PointCloud<int> keypoint_indices;
    narf_keypoint_detector.compute (keypoint_indices);
    std::cout << "Found "<<keypoint_indices.points.size ()<<" key points.\n";

    keypoints_out->points.resize (keypoint_indices.points.size());
    for (size_t i=0; i<keypoint_indices.points.size(); ++i)
        keypoints_out->points[i].getVector3fMap() = range_image.points[keypoint_indices.points[i]].getVector3fMap();

}


void
compute_PFH_features_at_keypoints (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points,
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                   pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, float feature_radius,
                                   pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr &descriptors_out)
{
  // Create a PFHRGBEstimation object
  pcl::PFHRGBEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::PFHRGBSignature250> pfh_est;

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  pfh_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGBA>));

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch (feature_radius);

  /* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
   * use them as an input to our PFH estimation, which expects clouds of PointXYZRGBA points.  To get around this,
   * we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to
   * "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGBA>).  Note that the original cloud doesn't have any RGB
   * values, so when we copy from PointWithScale to PointXYZRGBA, the new r,g,b fields will all be zero.
   */

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::copyPointCloud (*keypoints, *keypoints_xyzrgb);

  // Use all of the points for analyzing the local structure of the cloud
  pfh_est.setSearchSurface (points);
  pfh_est.setInputNormals (normals);

  // But only compute features at the keypoints
  pfh_est.setInputCloud (keypoints_xyzrgb);

  // Compute the features
  pfh_est.compute (*descriptors_out);
}

void
compute_FPFH_features_at_keypoints (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &points,
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                   pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, float feature_radius,
                                   pcl::PointCloud<pcl::FPFHSignature33>::Ptr &descriptors_out)
{
  // Create a FPFHEstimation object
  pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh_est;

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  fpfh_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGBA>));

  // Specify the radius of the PFH feature
  fpfh_est.setRadiusSearch (feature_radius);

  /* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
   * use them as an input to our PFH estimation, which expects clouds of PointXYZRGBA points.  To get around this,
   * we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to
   * "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGBA>).  Note that the original cloud doesn't have any RGB
   * values, so when we copy from PointWithScale to PointXYZRGBA, the new r,g,b fields will all be zero.
   */

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::copyPointCloud (*keypoints, *keypoints_xyzrgb);

  // Use all of the points for analyzing the local structure of the cloud
  fpfh_est.setSearchSurface (points);
  fpfh_est.setInputNormals (normals);

  // But only compute features at the keypoints
  fpfh_est.setInputCloud (keypoints_xyzrgb);

  // Compute the features
  fpfh_est.compute (*descriptors_out);
}

void
find_pfh_feature_correspondences (pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{
  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  pcl::search::KdTree<pcl::PFHRGBSignature250> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
  }
}

void
find_fpfh_feature_correspondences (pcl::PointCloud<pcl::FPFHSignature33>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::FPFHSignature33>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{
  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  pcl::search::KdTree<pcl::FPFHSignature33> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
  }
}

void printKeypoints(const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints)
{

    for (size_t i = 0; i < keypoints->size (); ++i)
    {
        /** Get the point data **/
        const pcl::PointWithScale & p = keypoints->points[i];

        std::cout<<"KEYPOINT: "<<p.x<<" "<<p.y<<" "<<p.z<<"\n";
    }

    return;
}

void visualize_normals (const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points,
                        const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr normal_points,
                        const pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  // Add the points and normals to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points, "points");
  viz.addPointCloud (normal_points, "normal_points");
  std::cout<<"points->size(): "<<points->size()<<"\n";
  std::cout<<"normal_points->size(): "<<normal_points->size()<<"\n";

  viz.addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (normal_points, normals, 1, 0.03, "normals");

  // Give control over to the visualizer
  viz.spin ();
}

void visualize_keypoints (const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points,
                          const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints)
{
  // Add the points to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  //viz.setBackgroundColor (1, 1, 1);
  viz.addPointCloud (points, "points");

  // Draw each keypoint as a sphere
  for (size_t i = 0; i < keypoints->size (); ++i)
  {
    // Get the point data
    const pcl::PointWithScale & p = keypoints->points[i];

    // Pick the radius of the sphere *
    float r = 2 * p.scale;
    // * Note: the scale is given as the standard deviation of a Gaussian blur, so a
    //   radius of 2*p.scale is a good illustration of the extent of the keypoint

    // Generate a unique string for each sphere
    std::stringstream ss ("keypoint");
    ss << i;

    // Add a sphere at the keypoint
    viz.addSphere (p, r, 1.0, 0.0, 0.0, ss.str ());
  }

  // Give control over to the visualizer
  viz.spin ();
}

void visualize_correspondences (const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points1,
                                const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1,
                                const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points2,
                                const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2,
                                const std::vector<int> &correspondences,
                                const std::vector<float> &correspondence_scores)
{
  // We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
  // by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points

  // Create some new point clouds to hold our transformed data
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points_left (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_left (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points_right (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_right (new pcl::PointCloud<pcl::PointWithScale>);

  // Shift the first clouds' points to the left
  //const Eigen::Vector3f translate (0.0, 0.0, 0.3);
  const Eigen::Vector3f translate (3.0, 0.0, 0.0);
  const Eigen::Quaternionf no_rotation (0, 0, 0, 0);
  pcl::transformPointCloud (*points1, *points_left, -translate, no_rotation);
  pcl::transformPointCloud (*keypoints1, *keypoints_left, -translate, no_rotation);

  // Shift the second clouds' points to the right
  pcl::transformPointCloud (*points2, *points_right, translate, no_rotation);
  pcl::transformPointCloud (*keypoints2, *keypoints_right, translate, no_rotation);

  // Add the clouds to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points_left, "points_left");
  viz.addPointCloud (points_right, "points_right");

  // Compute the median correspondence score
  std::vector<float> temp (correspondence_scores);
  std::sort (temp.begin (), temp.end ());
  float median_score = temp[temp.size ()/2];

  for (size_t i = 0; i < temp.size(); ++i)
  {
    std::cout<<"Sort Score distance: "<<temp[i]<<"\n";
  }

  float percentage = 1.0;
  //if (temp.size() > 1.0)
  //  percentage = 0.8;

  std::cout<<"Median Score distance: "<<median_score<<"\n";
  // Draw lines between the best corresponding points
  for (size_t i = 0; i < keypoints_left->size (); ++i)
  {
    std::cout<<"Score distance: "<<correspondence_scores[i]<<"\n";

    if (correspondence_scores[i] > percentage * median_score)
    {
      continue; // Don't draw weak correspondences
    }

    // Get the pair of points
    const pcl::PointWithScale & p_left = keypoints_left->points[i];
    const pcl::PointWithScale & p_right = keypoints_right->points[correspondences[i]];

    // Generate a random (bright) color
    double r = (rand() % 100);
    double g = (rand() % 100);
    double b = (rand() % 100);
    double max_channel = std::max (r, std::max (g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;

    // Generate a unique string for each line
    std::stringstream ss ("line");
    ss << i;

    // Draw the line
    viz.addLine (p_left, p_right, r, g, b, ss.str ());
  }

  // Give control over to the visualizer
  viz.spin ();
}

int normals_demo (const char * filename, const char *radius)
{
  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

  // Load a point cloud
  pcl::io::loadPCDFile (filename, *points);

  // Downsample the cloud
  const float voxel_grid_leaf_size = 0.05;
  downsample (points, voxel_grid_leaf_size, downsampled);
  //pcl::copyPointCloud(*points, *downsampled);

  // Compute surface normals
  const float normal_radius = std::stof(radius);
  compute_surface_normals (downsampled, normal_radius, normals);

  // Visualize the points and normals
  visualize_normals (points, downsampled, normals);

  return (0);
}

int keypoints_demo (const char * filename,
            const float min_scale,
            const int nr_octaves,
            const int nr_octaves_per_scale,
            const float min_contrast)
{

  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints (new pcl::PointCloud<pcl::PointWithScale>);

  // Load a point cloud
  std::cerr<<"LOADING FILE... [";
  pcl::io::loadPCDFile (filename, *points);

  std::cerr<<"FILE LOAD SUCCESSFULLY]\n";

  std::cout<<"min scale: "<<min_scale<<"\n";
  std::cout<<"nr_octaves: "<<nr_octaves<<"\n";
  std::cout<<"nr_octaves_per_scale: "<<nr_octaves_per_scale<<"\n";
  std::cout<<"min_contrast: "<<min_contrast<<"\n";

  // Compute keypoints
  detect_keypoints (points, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints);
  printKeypoints(keypoints);
  std::cerr<<"DETECTED "<<keypoints->size()<<" keypoints\n";

  // Visualize the point cloud and its keypoints
  visualize_keypoints (points, keypoints);

  return (0);
}

int correspondences_demo_pfh (const char * filename_base, const char* number_one, const char* number_two)
{
  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr descriptors1 (new pcl::PointCloud<pcl::PFHRGBSignature250>);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr descriptors2 (new pcl::PointCloud<pcl::PFHRGBSignature250>);

  // Load the pair of point clouds
  std::stringstream ss1, ss2;
  ss1 << filename_base<< number_one << ".pcd";
  pcl::io::loadPCDFile (ss1.str (), *points1);
  ss2 << filename_base<< number_two << ".pcd";
  pcl::io::loadPCDFile (ss2.str (), *points2);

  std::cout<<"PFH\n";
  std::cout<<"PCD 1: "<<ss1.str()<<"\n";
  std::cout<<"PCD 2: "<<ss2.str()<<"\n";

  // Downsample the cloud
  const float voxel_grid_leaf_size = 0.05;
  downsample (points1, voxel_grid_leaf_size, downsampled1);
  downsample (points2, voxel_grid_leaf_size, downsampled2);

  // Compute surface normals
  const float normal_radius = 0.1;
  compute_surface_normals (downsampled1, normal_radius, normals1);
  compute_surface_normals (downsampled2, normal_radius, normals2);

  // Compute keypoints
  const float min_scale = 0.08;
  const int nr_octaves = 10;
  const int nr_octaves_per_scale = 10;
  const float min_contrast = 5.0;
  detect_keypoints (points1, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints1);
  detect_keypoints (points2, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints2);

  // Compute PFH features
  const float feature_radius = 1.0;
  compute_PFH_features_at_keypoints (downsampled1, normals1, keypoints1, feature_radius, descriptors1);
  compute_PFH_features_at_keypoints (downsampled2, normals2, keypoints2, feature_radius, descriptors2);

  // Find feature correspondences
  std::vector<int> correspondences;
  std::vector<float> correspondence_scores;
  find_pfh_feature_correspondences (descriptors1, descriptors2, correspondences, correspondence_scores);

  // Print out ( number of keypoints / number of points )
  std::cout << "First cloud: Found " << keypoints1->size () << " keypoints "
            << "out of " << downsampled1->size () << " total points." << std::endl;
  std::cout << "Second cloud: Found " << keypoints2->size () << " keypoints "
            << "out of " << downsampled2->size () << " total points." << std::endl;

  // Visualize the two point clouds and their feature correspondences
  visualize_correspondences (points1, keypoints1, points2, keypoints2, correspondences, correspondence_scores);

  return (0);
}

int correspondences_demo_fpfh (const char * filename_base, const char* number_one, const char* number_two)
{
  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors1 (new pcl::PointCloud<pcl::FPFHSignature33>);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors2 (new pcl::PointCloud<pcl::FPFHSignature33>);

  // Load the pair of point clouds
  std::stringstream ss1, ss2;
  ss1 << filename_base<< number_one << ".pcd";
  pcl::io::loadPCDFile (ss1.str (), *points1);
  ss2 << filename_base<< number_two << ".pcd";
  pcl::io::loadPCDFile (ss2.str (), *points2);

  std::cout<<"FAST PFH\n";
  std::cout<<"PCD 1: "<<ss1.str()<<"\n";
  std::cout<<"PCD 2: "<<ss2.str()<<"\n";

  // Downsample the cloud
  const float voxel_grid_leaf_size = 0.05;
  downsample (points1, voxel_grid_leaf_size, downsampled1);
  downsample (points2, voxel_grid_leaf_size, downsampled2);

  // Compute surface normals
  const float normal_radius = 0.1;
  compute_surface_normals (downsampled1, normal_radius, normals1);
  compute_surface_normals (downsampled2, normal_radius, normals2);

  // Compute keypoints
  const float min_scale = 0.10;
  const int nr_octaves = 10;
  const int nr_octaves_per_scale = 10;
  const float min_contrast = 3;
  detect_keypoints (points1, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints1);
  detect_keypoints (points2, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints2);

  std::cout<<"min scale: "<<min_scale<<"\n";
  std::cout<<"nr_octaves: "<<nr_octaves<<"\n";
  std::cout<<"nr_octaves_per_scale: "<<nr_octaves_per_scale<<"\n";
  std::cout<<"min_contrast: "<<min_contrast<<"\n";

  std::cout<<"KEYPOINTS 1:\n";
  printKeypoints(keypoints1);
  std::cout<<"KEYPOINTS 2:\n";
  printKeypoints(keypoints2);

  // Compute PFH features
  const float feature_radius = 1.0;
  compute_FPFH_features_at_keypoints (downsampled1, normals1, keypoints1, feature_radius, descriptors1);
  compute_FPFH_features_at_keypoints (downsampled2, normals2, keypoints2, feature_radius, descriptors2);

  // Find feature correspondences
  std::vector<int> correspondences;
  std::vector<float> correspondence_scores;
  find_fpfh_feature_correspondences (descriptors1, descriptors2, correspondences, correspondence_scores);

  // Print out ( number of keypoints / number of points )
  std::cout << "First cloud: Found " << keypoints1->size () << " keypoints "
            << "out of " << downsampled1->size () << " total points." << std::endl;
  std::cout << "Second cloud: Found " << keypoints2->size () << " keypoints "
            << "out of " << downsampled2->size () << " total points." << std::endl;

  std::cout << "FOUND: "<<correspondences.size()<<" CORRESPONDENCES\n";

  // Visualize the two point clouds and their feature correspondences
  visualize_correspondences (points1, keypoints1, points2, keypoints2, correspondences, correspondence_scores);

  return (0);
}

int correspondences_demo_narf (const char * filename_base, const char* number_one, const char* number_two)
{
  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors1 (new pcl::PointCloud<pcl::FPFHSignature33>);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors2 (new pcl::PointCloud<pcl::FPFHSignature33>);

  // Load the pair of point clouds
  std::stringstream ss1, ss2;
  ss1 << filename_base<< number_one << ".pcd";
  pcl::io::loadPCDFile (ss1.str (), *points1);
  ss2 << filename_base<< number_two << ".pcd";
  pcl::io::loadPCDFile (ss2.str (), *points2);

  std::cout<<"NARF\n";
  std::cout<<"PCD 1: "<<ss1.str()<<" with "<<points1->size()<<" points\n";
  std::cout<<"PCD 2: "<<ss2.str()<<" with "<<points2->size()<<" points\n";

  // Downsample the cloud
  const float voxel_grid_leaf_size = 0.05;
  downsample (points1, voxel_grid_leaf_size, downsampled1);
  downsample (points2, voxel_grid_leaf_size, downsampled2);

  // Compute surface normals
  const float normal_radius = 0.1;
  compute_surface_normals (downsampled1, normal_radius, normals1);
  compute_surface_normals (downsampled2, normal_radius, normals2);

  // Compute keypoints
  float angular_resolution = 0.5f;
  float noise_level = 0.4;
  float min_range = 0.0f;
  int border_size = 1;
  float support_size = 0.2f;
  detect_narf_keypoints (points1, angular_resolution, noise_level, min_range, border_size, support_size, keypoints1);
  detect_narf_keypoints (points2, angular_resolution, noise_level, min_range, border_size, support_size, keypoints2);

  // Compute PFH features
//  const float feature_radius = 1.0;
//  compute_narf_features_at_keypoints (downsampled1, normals1, keypoints1, feature_radius, descriptors1);
//  compute_narf_features_at_keypoints (downsampled2, normals2, keypoints2, feature_radius, descriptors2);
//
//  // Find feature correspondences
//  std::vector<int> correspondences;
//  std::vector<float> correspondence_scores;
//  find_narf_feature_correspondences (descriptors1, descriptors2, correspondences, correspondence_scores);
//
//  // Print out ( number of keypoints / number of points )
//  std::cout << "First cloud: Found " << keypoints1->size () << " keypoints "
//            << "out of " << downsampled1->size () << " total points." << std::endl;
//  std::cout << "Second cloud: Found " << keypoints2->size () << " keypoints "
//            << "out of " << downsampled2->size () << " total points." << std::endl;
//
//  // Visualize the two point clouds and their feature correspondences
//  visualize_correspondences (points1, keypoints1, points2, keypoints2, correspondences, correspondence_scores);
//
  return (0);
}

int main (int argc, char ** argv)
{
  if (argc < 3)
  {
    std::cout << "Please pass a filename and exactly one of the following arguments: "
              << "normals, keypoints, correspondences" << std::endl;
    return (-1);
  }

  if (strcmp (argv[2], "normals") == 0)
  {
    std::cout<<"Normals...\n";
    return (normals_demo (argv[1], argv[3]));
  }
  else if (strcmp (argv[2], "keypoints") == 0)
  {
    std::cout<<"Keypoints...\n";
    return (keypoints_demo (argv[1], std::stof(argv[3]), std::stof(argv[4]), std::stof(argv[5]), std::stof(argv[6])));
  }
  else if (strcmp (argv[2], "correspondences") == 0)
  {
    std::cout<<"Correspondences...\n";
    if (strcmp (argv[3], "1") == 0)
    {
        return (correspondences_demo_pfh (argv[1], argv[4], argv[5]));
    }
    else if (strcmp (argv[3], "2") == 0)
    {
        return (correspondences_demo_fpfh (argv[1], argv[4], argv[5]));
    }
    else if (strcmp (argv[3], "3") == 0)
    {
        return (correspondences_demo_narf (argv[1], argv[4], argv[5]));
    }

  }
}

