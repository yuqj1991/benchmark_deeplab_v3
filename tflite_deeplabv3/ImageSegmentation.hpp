#pragma once
#include <opencv2/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif 
#define USE_GL_DELEGATE

#include "tensorflow/lite/c/c_api.h"

#if defined(USE_GPU_DELEGATEV2)
	#include "tensorflow/lite/delegates/gpu/delegate.h"
#elif defined(USE_GL_DELEGATE)
	#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

using namespace std;
using namespace cv;

struct SegmentationResult
{
	float segmentedArea;
	Mat mask;

	SegmentationResult(int segmentedArea, Mat mask)
	{
		SegmentationResult::segmentedArea = segmentedArea;
		SegmentationResult::mask = mask;
	}
};

class ImageSegmentation
{
public:
	// Methods
	ImageSegmentation(const char* deeplabModelPath, bool quantized = false);
	~ImageSegmentation();
	bool segmentImage(Mat src, Mat& result);
private:
	// Members
	const int MODEL_SIZE = 257;
	const int MODEL_CNLS = 3;
	const int CLASS_COUNT = 21;
	const float IMAGE_MEAN = 128.0;
	const float IMAGE_STD = 128.0;
	bool m_modelQuantized = false;
	TfLiteModel* m_model;
	TfLiteInterpreter* m_interpreter;
	TfLiteTensor* m_input_tensor = nullptr;
	const TfLiteTensor* m_output_mask = nullptr;

	TfLiteDelegate *m_pDelegate = nullptr;

	// Methods
	void initModel(const char* deeplabModelPath);
};
