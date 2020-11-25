
#ifndef _TFLITE_SEGMENT_H
#define _TFLITE_SEGMENT_H

#include "opencv2/core.hpp"

#if defined(_MSC_VER)
	#ifdef ALGORITHMLIB_EXPORTS
		#define SEGMENT_EXPORTS   __declspec(dllexport)
	#else
		#define SEGMENT_EXPORTS   __declspec(dllimport)
	#endif
#elif defined(__GNUC__)
	#ifdef ALGORITHMLIB_EXPORTS
		#define SEGMENT_EXPORTS   __attribute__((visibility("default")))
	#else
		#define SEGMENT_EXPORTS
	#endif
#else
	#pragma warning Unknown dynamic link import/export semantics.
#endif
	
	SEGMENT_EXPORTS void segmentMat(Mat src, Mat& mask);
	SEGMENT_EXPORTS bool segmentPath(const char *pchSrcPath,const char *pchMaskPath);
#endif