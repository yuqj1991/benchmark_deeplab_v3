#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ImageSegmentation.hpp"
#include <chrono>
#include "tflite_segment.h"

using namespace std;
using namespace cv;
using namespace std::chrono;




#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#if 1
int make_folder(const char* path)
{
    DIR *dp;
    int status = 0;
    if ((dp = opendir(path)) == NULL)
    {
        printf("%s is not a directory or not exist!, have already make new one dir automaticly\n", path);
        int flag=mkdir(path, 0777);
        if(flag == 0){
            status = 0;
        }else{
            status = -1;
        }
    }else{
        printf("%s is a directory && exist!\n", path);
        status = 0;
    }
    
    closedir(dp);
    return status;
}
std::vector<std::pair<std::string, std::string> > get_image(const std::string & folder_name){
    struct dirent *faceSetDir;
    DIR* dir = opendir(folder_name.c_str());
    std::vector<std::pair<std::string, std::string> > fullImage;
    if( dir == NULL )
        printf(" is not a directory or not exist!\n");
    while ((faceSetDir = readdir(dir)) != NULL) {
        if(strcmp(faceSetDir->d_name,".")==0 || strcmp(faceSetDir->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(faceSetDir->d_name[0] == '.')
            continue;
        else if (faceSetDir->d_type == DT_REG) {
            std::string newImagefile = folder_name + string("/") + string(faceSetDir->d_name);
            std::string imagename = faceSetDir->d_name;
            fullImage.push_back(std::pair<std::string, std::string>(newImagefile, imagename));
        }
    }
    closedir(dir);
    return fullImage;
}

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

double segmentMat(Mat src, Mat& mask, ImageSegmentation &g_segmentation )
{
    double tt = get_current_time();
	g_segmentation.segmentImage(src, mask);
	double te = get_current_time();
	return double(te - tt);
}

bool segmentPath(const char *pchSrcPath,const char *pchMaskPath, const char *modelPath)
{
    ImageSegmentation g_segmentation = ImageSegmentation(modelPath);
    std::vector<std::pair<std::string, std::string> > images = get_image(pchSrcPath);
    bool bRet = true;
    double time = 0;
    int image_count = 0;
    for(unsigned int ii = 0; ii < images.size(); ii++){
        cv::Mat src = cv::imread(images[ii].first);
	    cv::Mat mask;

        time += segmentMat(src, mask, g_segmentation);
        if (mask.empty())
		    bRet = false;
	    else{
            int pos=images[ii].second.find(".jpg");
            std::string outName = string(pchMaskPath) + string("/") + images[ii].second.substr(0, pos) + std::string(".png");
            cv::imwrite(outName.c_str(), mask);
        }
        image_count++;
        if(time > 10000){
            break;
        }		    
    }
	
	return bRet;
}

int main(int argc, const char* argv[])
{
    if(argc != 2){
        printf("./useage: ./deeplabv3_test model.tflite\n");
        return 0;
    }
	segmentPath("../../images", "../../output", argv[1]);
    return 0;
}
