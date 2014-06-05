#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/opencv.c"
#else

//======================================================================
// File: opencv
//
// Description: A wrapper for a couple of OpenCV functions.
//
// Created: February 12, 2010, 1:22AM
//
// Install on ubuntu :
//  http://www.samontab.com/web/2010/04/installing-opencv-2-1-in-ubuntu/
//
// Author: Clement Farabet // clement.farabet@gmail.com
//         Marco Scoffier // github@metm.org (more functions)
//======================================================================

#include <luaT.h>
#include <TH.h>

#include <dirent.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <signal.h>
#include <pthread.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv_backcomp.hpp"

#define CV_NO_BACKWARD_COMPATIBILITY


static int libopencv_(Main_iplType)() {
  
  #if defined(TH_REAL_IS_BYTE)
    return IPL_DEPTH_8U;
  #elif defined(TH_REAL_IS_CHAR)
    return IPL_DEPTH_8S;
  #elif defined(TH_REAL_IS_SHORT)    
    return IPL_DEPTH_16S;
  #elif defined(TH_REAL_IS_LONG)        
    return IPL_DEPTH_32S;
  #elif defined(TH_REAL_IS_FLOAT)        
    return IPL_DEPTH_32F;
  #elif defined(TH_REAL_IS_DOUBLE)    
    printf("%d???\n", sizeof(real));
    return IPL_DEPTH_64F;
  #endif
      
  THError("iplType: bad torch type");
}


static IplImage * libopencv_(Main_torch2cv_rgb)(THTensor *source) {
  
  // Pointers
  real* dest_data = NULL;

  // Get size and channels
  int channels = source->size[0];
  int dest_step;
  CvSize dest_size = cvSize(source->size[2], source->size[1]);

  int type = libopencv_(Main_iplType)();

  
  // Create ipl image
  IplImage * dest = cvCreateImage(dest_size, type, channels);

  // get pointer to raw data
  cvGetRawData(dest, (uchar**)&dest_data, &dest_step, &dest_size);

  // copy and transpose
  THTensor *transposed = THTensor_(newTranspose)(source, 0, 2);
  THTensor_(transpose)(transposed, transposed, 0, 1);
  
  THTensor *tensor = THTensor_(newContiguous)(transposed);
  
  real *source_data = THTensor_(data)(tensor);
  size_t size = THTensor_(nElement)(tensor) * sizeof(real);
   
  memcpy(dest_data, source_data, size);
  
  // free
  THTensor_(free)(tensor);
  THTensor_(free)(transposed);
  
  // return freshly created IPL image
  return dest;
}




static void libopencv_(Main_opencvPoints2torch)(CvPoint2D32f * points, int npoints, THTensor *tensor) {

  // Resize target
  THTensor_(resize2d)(tensor, npoints, 2);
  THTensor *tensorc = THTensor_(newContiguous)(tensor);
  real *tensor_data = THTensor_(data)(tensorc);

  // copy
  int p;
  for (p=0; p<npoints; p++){
    *tensor_data++ = (real)points[p].x;
    *tensor_data++ = (real)points[p].y;
  }

  // cleanup
  THTensor_(free)(tensorc);
}

static CvPoint2D32f * libopencv_(Main_torch2opencvPoints)(THTensor *src) {

  int count = src->size[0];
  int dims  = src->size[1];
  // create output
  CvPoint2D32f * points_cv = NULL;
  points_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points_cv[0]));
  real * src_pt = THTensor_(data)(src);
  // copy
  int p,q;
  for (p=0; p<count; p++){
    points_cv[p].x = (float)*src_pt++ ;
    points_cv[p].y = (float)*src_pt++ ;
    if (dims > 2) {
      for (q=2 ; q<dims ; q++){
        src_pt++;
      }
    }
  }

  // return freshly created CvPoint2D32f
  return points_cv;
}


//============================================================
static int libopencv_(Main_cvResize) (lua_State *L) {
  // Get Tensor's Info
//   THTensor * source = luaT_checkudata(L, 1, torch_Tensor);
//   THTensor * dest = luaT_checkudata(L, 2, torch_Tensor);  
// 
//   int quality = lua_tonumber(L, 3);
//   
//   // Generate IPL headers
//   IplImage * source_ipl = libopencv_(Main_torchimg2opencv_8U)(source);
//     
//   CvSize dest_size = cvSize(dest->size[2], dest->size[1]);
//   IplImage * dest_ipl = cvCreateImage(dest_size, IPL_DEPTH_8U, dest->size[0]);
//   
//   int flags = qualityFlags(quality);  
//   
//   // Simple call to CV function
//   cvResize(source_ipl, dest_ipl, flags);
// 
//   // Return results
//   libopencv_(Main_opencv8U2torch)(dest_ipl, dest);
// 
//   // Deallocate headers
//   cvReleaseImageHeader(&source_ipl);
//   cvReleaseImageHeader(&dest_ipl);

  return 0;
}


//============================================================
static int libopencv_(Main_cvWarpAffine) (lua_State *L) {
  // Get Tensor's Info
//   THTensor * source = luaT_checkudata(L, 1, torch_Tensor);
//   THTensor * dest   = luaT_checkudata(L, 2, torch_Tensor);
//   THDoubleTensor * warp   = luaT_checkudata(L, 3, "torch.DoubleTensor");
//   
//   int quality = lua_tonumber(L, 4);
//   int fill = lua_toboolean(L, 5);
// 
//   THArgCheck(warp->size[0] == 2 , 1, "warp matrix: 2x3 Tensor expected");
//   THArgCheck(warp->size[1] == 3 , 1, "warp matrix: 2x3 Tensor expected");
// 
//   // Generate IPL headers
// 
//   IplImage * source_ipl = libopencv_(Main_torchimg2opencv_8U)(source);
//   IplImage * dest_ipl = libopencv_(Main_torchimg2opencv_8U)(dest);
//   
//   
//   cvStartWindowThread();
//   
// 
//   
// //   CvScalar mean = cvAvg(source_ipl, NULL);
// //   printf("%f %f %f %f \n", mean.val[0], mean.val[1],mean.val[2],mean.val[3]);
//   
//   
//   CvMat* warp_mat = cvCreateMat(2,3,CV_32FC1);
// 
//   // Copy warp transformation matrix
//   THDoubleTensor *tensor = THDoubleTensor_newContiguous(warp);
//   float* ptr = warp_mat->data.fl;
//   TH_TENSOR_APPLY(double, tensor,
//                   *ptr = (float)*tensor_data;
//                   ptr++;
//                   );
//   THDoubleTensor_free(tensor);
//   
//   int flags = (fill ? CV_WARP_FILL_OUTLIERS : 0) | qualityFlags(quality);  
//   
//   // Simple call to CV function
//   cvWarpAffine(source_ipl, dest_ipl, warp_mat,
//                 flags, cvScalarAll(0));
//   
//   cvNamedWindow("booho", CV_WINDOW_AUTOSIZE);Main_torchimg2opencv_8U
//   cvShowImage("booho", dest_ipl);
// 
//   // Return results
//   libopencv_(Main_opencv8U2torch)(dest_ipl, dest);
//   
//   
// 
//   // Deallocate headers
//   cvReleaseImageHeader(&source_ipl);
//   cvReleaseImageHeader(&dest_ipl);
//   cvReleaseMat( &warp_mat );

  return 0;
}

//============================================================
static int libopencv_(Main_cvGetAffineTransform) (lua_State *L) {
  // Get Tensor's Info
//   THTensor * src = luaT_checkudata(L, 1, torch_Tensor);
//   THTensor * dst   = luaT_checkudata(L, 2, torch_Tensor);
//   THTensor * warp   = luaT_checkudata(L, 3, torch_Tensor);
// 
//   /* THArgCheck(src->size[0] == 3 , 1, "input: 2x3 Tensor expected"); */
//   /* THArgCheck(src->size[1] == 2 , 1, "input: 2x3 Tensor expected"); */
//   /* THArgCheck(dst->size[0] == 3 , 2, "input: 2x3 Tensor expected"); */
//   /* THArgCheck(dst->size[1] == 2 , 2, "input: 2x3 Tensor expected"); */
// 
//   CvPoint2D32f* srcTri = libopencv_(Main_torch2opencvPoints)(src);
//   CvPoint2D32f* dstTri = libopencv_(Main_torch2opencvPoints)(dst);
// 
//   CvMat* warp_mat = cvCreateMat(2,3,CV_32FC1);
//   cvGetAffineTransform( srcTri, dstTri, warp_mat );
// 
//   libopencv_(Main_opencvMat2torch)(warp_mat,warp);
// 
//   // Deallocate headers
//   cvFree(&srcTri);
//   cvFree(&dstTri);
//   cvReleaseMat(&warp_mat);

  return 0;
}



static int libopencv_(cvDisplay)(lua_State* L) {
  
  THTensor * image    = luaT_checkudata(L, 1, torch_Tensor);
  IplImage * image_cv = libopencv_(Main_torch2cv_rgb)(image);
  printf("%d \n", image_cv->depth);
  
  cvCvtColor(image_cv, image_cv, CV_BGR2RGB);
  
  cvStartWindowThread();

  const char* win     = lua_tostring(L, 2);  
  cvNamedWindow(win, CV_WINDOW_AUTOSIZE);
  cvShowImage(win, image_cv);

  cvWaitKey(0);

  cvReleaseImage(&image_cv);
  return 0;
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libopencv_(Main__) [] =
{
  {"getAffineTransform",   libopencv_(Main_cvGetAffineTransform)},
  {"warpAffine",           libopencv_(Main_cvWarpAffine)},
  {"resize",               libopencv_(Main_cvResize)},
  {"display",              libopencv_(cvDisplay)},
  {NULL, NULL}  /* sentinel */
};




DLL_EXPORT int libopencv_(Main_init) (lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, libopencv_(Main__), "libopencv");
  lua_pop(L,1); 
  return 1;
}

#endif
