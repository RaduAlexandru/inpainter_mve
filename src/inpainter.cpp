#include <iostream>
#include <algorithm>
#include <iterator>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include "mve/image_io.h"
#include "mve/image_base.h"
#include "mve/image.h"
#include "mve/defines.h"

#include "domain_transform.h"

#include <boost/filesystem.hpp>
#include <boost/iterator/filter_iterator.hpp>
namespace fs = boost::filesystem;


float  interpolate(float value, float leftMin, float leftMax, float rightMin, float rightMax){
    float leftSpan = leftMax - leftMin;
    float rightSpan = rightMax - rightMin;

    float valueScaled = (value - leftMin) / (leftSpan);

    float value_to_return= rightMin + (valueScaled * rightSpan);
    // std::cout << "value to return" << value_to_return << std::endl;
    return value_to_return;
}


void show_img (cv::Mat img){
  cv::Mat copy;
  img.copyTo(copy);
  double min;
  double max;
  cv::minMaxIdx(img, &min, &max);
  std::cout << "showimg minmax is" << min << " " << max << std::endl;

  cv::Mat adjMap;

  for (size_t i = 0; i < copy.rows; i++) {
     for (size_t j = 0; j < copy.cols; j++) {
       copy.at<float>(i,j)= interpolate (copy.at<float>(i,j), min, max, 0.0, 1.0);
     }
   }

  cv::imshow("Out",copy);
  cv::waitKey(0); //wait infinite time for a keypress

}


int main( int argc, char** argv )
{

  std::cout << "starting processing" << std::endl;

  if (argc!=3){
    std::cout << "Invalid number of arguments" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "\t inpainter -L<level> <scene-dir>"  << std::endl << std::endl;
    std::cout << "Level: Indicates the downsample level of the depthmap, 0 being the original size" << std::endl;
    return 2;
  }

  std::string level= argv[1];
  std::string scene_path= argv[2];

  level=level.back();

  std::cout << "level is "<< level << std::endl;


  //get the path
  std::string path= scene_path + "/views/";
  fs::path path_boost(path);
  fs::path absPath = fs::absolute(path_boost);
  fs::path full_p = fs::canonical(absPath);
  path=full_p.string();
  std::cout << "path is " << path << std::endl;

  //get number of views in the scene
  fs::path p(path);
  fs::directory_iterator dir_first(p), dir_last;
  auto pred = [](const fs::directory_entry& p)
  {
      return fs::is_directory(p);
  };
  int num_views = std::distance(boost::make_filter_iterator(pred, dir_first, dir_last),
                  boost::make_filter_iterator(pred, dir_last, dir_last));


  std::cout << "num views" << num_views << std::endl;



  for (size_t i = 0; i < num_views; i++) {
    std::ostringstream num_str;
    num_str <<  std::setfill('0') << std::setw(4) << i << std::endl;
    std::string num_stripped;
    num_stripped=num_str.str();
    if (!num_stripped.empty() && num_stripped[num_stripped.length()-1] == '\n') {
    num_stripped.erase(num_stripped.length()-1);
    }

    std::string depth_name="depth-L" + level +".mvei";
    std::string img_path=path +  "/view_" + num_stripped+ ".mve/" + depth_name;
    std::cout << "dm img_path is" << img_path << std::endl;


    mve::FloatImage::Ptr dm_mve_float;     //img_mve_float = mve::FloatImage::create(cols, rows, channels);
    dm_mve_float = std::dynamic_pointer_cast<mve::FloatImage>(mve::image::load_mvei_file(img_path));

    int cols=dm_mve_float->width();
    int rows=dm_mve_float->height();
    int channels=dm_mve_float->channels();

    std::cout << "cols is " << cols << std::endl;
    std::cout << "rows is " << rows << std::endl;
    std::cout << "channels is " << channels << std::endl;


    //READ DM
    cv::Mat dm_cv(rows, cols, CV_32FC1); //The depth map to be read from file
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        dm_cv.at<float>(i,j)=dm_mve_float->at(j,i,0);
      }
    }


    //READ img
    // std::string img_name="undist-L2.png";
    std::string img_name="undistorted_original.png";
    // std::string img_name="undistorted.png";
    img_path=path +  "/view_" + num_stripped+ ".mve/"+ img_name;
    cv::Mat img_rgb= cv::imread(img_path);
    cv::Mat grayscale;
    cvtColor(img_rgb, grayscale, cv::COLOR_BGR2GRAY);
    //resize to be the size of depth
    cv::Size size(cols,rows);
    cv::resize(grayscale,grayscale,size);

    //convert to float
    grayscale.convertTo(grayscale,CV_32FC1);

    // cv::imshow("Display window",grayscale);
    // cv::waitKey(0);

    //Put nans int he depth where a 0 is present because domain transform requires it
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        if (dm_cv.at<float>(i,j)==0.0) {
          dm_cv.at<float>(i,j)= std::numeric_limits<float>::quiet_NaN();
        }
      }
    }




    //Fill depth
    depth_filler::DomainTransformFiller filter;
    cv::Mat dm_filled;
    dm_filled = filter.fillDepth(dm_cv,grayscale);



    //Show it
    // cv::normalize(dm_filled,dm_filled,0.0,1.0, cv::NORM_MINMAX, CV_32FC1);
    // cv::imshow("Display window",dm_filled);
    // cv::waitKey(0);


    //clamp some of the infinite values back to 0
    float min= std::numeric_limits<float>::max();
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        // std::cout << "pix dim filled is"  << dm_filled.at<float>(i,j) << std::endl;7
        if (dm_filled.at<float>(i,j)>=100) {
          dm_filled.at<float>(i,j)= 0.0f;
        }

        if (dm_filled.at<float>(i,j)<=0.1) {
          dm_filled.at<float>(i,j)= 0.0f;
        }

      }
    }



    //Save it as mvei
    mve::FloatImage::Ptr img_mve_float_save;       img_mve_float_save = mve::FloatImage::create(cols, rows, 1);

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        img_mve_float_save->at(j,i,0)=   dm_filled.at<float>(i,j);
      }
    }


    std::string save_path=path +  "/view_" + num_stripped+ ".mve/" + depth_name;
    mve::image::save_mvei_file(img_mve_float_save, save_path);


  }

  return 0;

}
