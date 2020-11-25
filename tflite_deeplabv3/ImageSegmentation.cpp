#include <chrono>
#include <algorithm>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "ImageSegmentation.hpp"

using namespace cv;

ImageSegmentation::ImageSegmentation(const char* deeplabModelPath, bool quantized)
{
	m_pDelegate = nullptr;
	m_modelQuantized = quantized;
	initModel(deeplabModelPath);
}

ImageSegmentation::~ImageSegmentation() {
	if (m_model != nullptr)
		TfLiteModelDelete(m_model);
	
#if defined (USE_GPU_DELEGATEV2)	
	if( m_pDelegate != nullptr)
		TfLiteGpuDelegateV2Delete(m_pDelegate);
#elif defined(USE_GPU_DELEGATE)
	if( m_pDelegate != nullptr)
		TfLiteGpuDelegateDelete(m_pDelegate);
#endif
}

void ImageSegmentation::initModel(const char* deeplabModelPath) {
	m_model = TfLiteModelCreateFromFile(deeplabModelPath);
	if (m_model == nullptr) {
		printf("Failed to load model");
		return;
	}

	// Build the interpreter
	TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 4);  // If using gpu, this is not related to inference speed.

	// Create gpu delegate
#if defined (USE_GL_DELEGATE)
    const TfLiteGpuDelegateOptions optDelegate = {
        .metadata = NULL,
        .compile_options = {
            .precision_loss_allowed = 1,  // FP16
            .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
            .dynamic_batch_enabled = 0,   // Not fully functional yet
			//.inline_parameters = 0,
        },
    };

    m_pDelegate = TfLiteGpuDelegateCreate(&optDelegate);

	if( m_pDelegate != nullptr ) {
		TfLiteInterpreterOptionsAddDelegate(options,m_pDelegate); 
		printf("Succeeded to create Gpu delegate!");
	}
	else{
		printf("Failed to create Gpu delegate!");
	}

#elif defined(USE_GPU_DELEGATEV2)
	TfLiteGpuDelegateOptionsV2 optDelegateV2 = TfLiteGpuDelegateOptionsV2Default();
    optDelegateV2.is_precision_loss_allowed = 1, // FP16
    optDelegateV2.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
	optDelegateV2.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
    optDelegateV2.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    optDelegateV2.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,

    m_pDelegate = TfLiteGpuDelegateV2Create(&optDelegateV2);
	// m_pDelegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
	if( m_pDelegate != nullptr ) {
		TfLiteInterpreterOptionsAddDelegate(options,m_pDelegate); 
		printf("Succeeded to create Gpu delegate!");
	}
	else{
		printf("Failed to create Gpu delegate!");
	}
#endif

	// Create the interpreter.
	m_interpreter = TfLiteInterpreterCreate(m_model, options);
	if (m_interpreter == nullptr) {
		printf("Failed to create interpreter");
		return;
	}

	
	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(m_interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	// Find input tensors.
	if (TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
	if (m_modelQuantized && m_input_tensor->type != kTfLiteUInt8) {
		printf("Detection model input should be kTfLiteUInt8!");
		return;
	}

	if (!m_modelQuantized && m_input_tensor->type != kTfLiteFloat32) {
		printf("Detection model input should be kTfLiteFloat32!");
		return;
	}

	if (m_input_tensor->dims->data[0] != 1 || m_input_tensor->dims->data[1] != MODEL_SIZE || m_input_tensor->dims->data[2] != MODEL_SIZE || m_input_tensor->dims->data[3] != MODEL_CNLS) {
		printf("Detection model must have input dims of 1x%ix%ix%i", MODEL_SIZE, MODEL_SIZE, MODEL_CNLS);
		return;
	}

	// Find output tensors.
	if (TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 output!");
		return;
	}

	m_output_mask = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);
}

bool ImageSegmentation::segmentImage(Mat src, Mat& result){
	if (m_model == nullptr) {
		return false;
	}

	int origWidth = src.cols;
	int origHeight = src.rows;

	int64 tt = getTickCount();

	Mat image;
	resize(src, image, Size(MODEL_SIZE, MODEL_SIZE), 0, 0, INTER_AREA);
	int cnls = image.type();
	if (cnls == CV_8UC1) {
		cvtColor(image, image, COLOR_GRAY2RGB);
	}
	else if (cnls == CV_8UC3) {
		cvtColor(image, image, COLOR_BGR2RGB);
	}
	else if (cnls == CV_8UC4) {
		cvtColor(image, image, COLOR_BGRA2RGB);
	}

	if (m_modelQuantized) {
		// Copy image into input tensor
		uchar* dst = m_input_tensor->data.uint8;
		memcpy(dst, image.data,
			sizeof(uchar) * MODEL_SIZE * MODEL_SIZE * MODEL_CNLS);
	}
	else {
		// Normalize the image based on std and mean (p' = (p-mean)/std)
		Mat fimage;
		image.convertTo(fimage, CV_32FC3, 1 / IMAGE_STD, -IMAGE_MEAN / IMAGE_STD);
		// Copy image into input tensor
		float* dst = m_input_tensor->data.f;
		memcpy(dst, fimage.data,
			sizeof(float) * MODEL_SIZE * MODEL_SIZE * MODEL_CNLS);
	}

	int64 te = getTickCount();
	cout << "Preprocessing time: "<< (double)(te - tt)/getTickFrequency() * 1000 << endl;

	tt = getTickCount();
	if (TfLiteInterpreterInvoke(m_interpreter) != kTfLiteOk) {
		printf("Error invoking detection model");
		return false;
	}

	te = getTickCount();
	cout << "Inference time: "<< (double)(te - tt)/getTickFrequency() * 1000 << endl;
	tt = te;

	const float* maskImage = m_output_mask->data.f;

	Mat mask = Mat(MODEL_SIZE, MODEL_SIZE, CV_8UC1, Scalar(0));
	unsigned char* maskData = mask.data;
	int segmentedPixels = 0;
	for (int y = 0; y < MODEL_SIZE; ++y) {
		for (int x = 0; x < MODEL_SIZE; ++x) {
			float max = -50;
			float cIndex = -1;
			for (int c = 0; c < 21 ; c++)
			{
				if (max < maskImage[y*MODEL_SIZE * 21 + x * 21 + c])
				{
					max = maskImage[y*MODEL_SIZE * 21 + x * 21 + c];
					cIndex = c;
				}
			}
			if (cIndex == 15 && max > 10)
				maskData[y * MODEL_SIZE + x] = 255;
		}
	}

	resize(mask, result, Size(origWidth, origHeight), 0, 0, INTER_CUBIC);

	te = getTickCount();
	cout << "Postprocessing time: "<< (double)(te - tt)/getTickFrequency() * 1000 << endl;

	return true;
}
