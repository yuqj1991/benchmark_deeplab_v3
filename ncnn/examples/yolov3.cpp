// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include <sys/time.h>
#ifdef SYSTEM_PROCESSOR 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#else
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif
#include <stdio.h>
#include <vector>

struct Object
{
    #ifdef SYSTEM_PROCESSOR
    cv::Rect_<float> rect;
    #else
    float x, y, width, height;
    #endif
    int label;
    float prob;
};
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#ifdef SYSTEM_PROCESSOR
static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects, const char *model_param, const char *model_bin)
{
    ncnn::Net yolov3;

    yolov3.opt.use_vulkan_compute = true;

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov3.load_param(model_param);
    yolov3.load_model(model_bin);


    const int target_size = 112;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"person"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}
#else
static int detect_yolov3(const unsigned char* bgr, int img_w, int img_h, std::vector<Object>& objects, 
                const char *model_param, const char *model_bin)
{
    ncnn::Net yolov3;

    yolov3.opt.use_vulkan_compute = true;

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov3.load_param(model_param);
    yolov3.load_model(model_bin);

    const int target_size = 112;


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR, img_w, img_h, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.x = values[2] * img_w;
        object.y = values[3] * img_h;
        object.width = values[4] * img_w - object.x;
        object.height = values[5] * img_h - object.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const std::vector<Object>& objects)
{
    static const char* class_names[] = {"person"};


    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.x, obj.y, obj.width, obj.height);

    }
}

static void SwitchValue(unsigned char & left, unsigned char &right){
    unsigned char temp = left;
    left = right;
    right = temp;
}
#endif
int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath] [model.param] [model.bin]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* model_param = argv[2];
    const char* model_bin = argv[3];
    #ifdef SYSTEM_PROCESSOR

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov3(m, objects, model_param, model_bin);

    draw_objects(m, objects);
    #else
    int width, height, cp;
    unsigned char *img = stbi_load(imagepath, &width, &height, &cp, 3); //RGB
    if(!img)
    {
        printf("The picture %s could not be loaded\n", imagepath);
        return 0;
    };
    for(int i = 0; i<height*width; i++)//toBGR
    {
        SwitchValue(img[i*3 + 0], img[i*3 + 2]);
    }

    std::vector<Object> objects;
    double time = 0., start = 0., end = 0.;
    int i = 0;
    while(i < 100)
    {
        start =  get_current_time();
        detect_yolov3(img, width, height, objects, model_param, model_bin);
        end = get_current_time();
        time += end - start;
        i++;
    } 
    printf("average time: %7.2f ms\n", time / 100);
    draw_objects(objects);
    #endif

    return 0;
}
