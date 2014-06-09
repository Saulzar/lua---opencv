#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/opencv.cpp"
#else


#include <luaT.h>
#include <TH.h>


#include <opencv2/opencv.hpp>




static int libopencv_(cvDepth)() {
  
  #if defined(TH_REAL_IS_BYTE)
    return CV_8U;
  #elif defined(TH_REAL_IS_CHAR)
    return CV_8S;
  #elif defined(TH_REAL_IS_SHORT)    
    return CV_16U;
  #elif defined(TH_REAL_IS_LONG)        
    return CV_32S;
  #elif defined(TH_REAL_IS_FLOAT)        
    return CV_32F;
  #elif defined(TH_REAL_IS_DOUBLE)    
    return CV_64F;
  #endif
      
  THError("cvType: bad torch type");
}



// template<typename T>
// inline void libopencv_(copyToMat)(cv::Mat_<T> &dest, THTensor *source) {
//   
//   // Get size and channels 
//   int channels = source->size[0];
//   int type = libopencv_(cvType)();
//   
//   if(channels != T::channels) {
//     THError("torch2cv: wrong number of channels in destination type");
//   }
//   
//   if(type != T::depth) {
//     THError("torch2cv: wrong type in destination type");
//   }
// 
//   // copy and transpose
//   THTensor *transposed = THTensor_(newTranspose)(source, 0, 2);
//   THTensor_(transpose)(transposed, transposed, 0, 1);
//   
//   THTensor *tensor = THTensor_(newContiguous)(transposed);
//   
//   real *data = THTensor_(data)(tensor);
//   size_t size = THTensor_(nElement)(tensor) * sizeof(real);
//   
//   cv::Mat_<T> result(source->size[1], source->size[2]);
// 
//   std::copy(data, data + size, dest.data(), (real*)result.data());
//   
//   // free
//   THTensor_(free)(tensor);
//   THTensor_(free)(transposed);
//   
//   // return freshly created IPL image
// }


inline bool libopencv_(CanConvert)(const char *from, cv::Mat const &m, THTensor *tensor) {
  
  int channels = tensor->size[2];
  int depth = libopencv_(cvDepth)();  
  
  
  assert (channels == m.channels() &&  "wrong number of channels");
  assert(depth == m.depth() && "wrong depth");
  assert(tensor->size[1] == m.rows && tensor->size[1] == m.cols && "wrong size");
  
  
  return true;
}

template<typename R>
inline R libopencv_(WithContiguous)(THTensor *source, std::function<R(THTensor *)> cont) {
  
  // copy and transpose
  THTensor *transposed = THTensor_(newTranspose)(source, 0, 2);
  THTensor_(transpose)(transposed, transposed, 0, 1);
  
  THTensor *tensor = THTensor_(newContiguous)(transposed);
  
  R r = cont(tensor);
  
  //
  // free
  THTensor_(free)(tensor);
  THTensor_(free)(transposed);
  
  return r;
}

// Blah blah
inline cv::Mat libopencv_(MakeMat)(THTensor *tensor, size_t width, size_t height, void *data = NULL) {
  int channels = tensor->size[2];
  int depth = libopencv_(cvDepth)();
  
  return cv::Mat(height, width, CV_MAKETYPE(depth, channels), data);
}


inline cv::Mat libopencv_(ContiguousToMat)(THTensor *tensor) {
  
  real *data = THTensor_(data)(tensor);  
  return libopencv_(MakeMat)(tensor, tensor->size[1], tensor->size[0], static_cast<void*>(data));
}

 

template<typename R>
inline R libopencv_(WithTensor)(THTensor *source, std::function<R(cv::Mat &)>  cont) {

  return libopencv_(WithContiguous)<R>(source, [cont](THTensor *tensor) {       
    cv::Mat result = libopencv_(ContiguousToMat)(tensor);
    return cont(result);
  });
  
}


inline cv::Mat libopencv_(ToMat)(THTensor *source) {
  
  return libopencv_(WithTensor)<cv::Mat>(source, [](cv::Mat &m) {
    return m.clone();
  });
}

inline THTensor *libopencv_(ToTensor)(cv::Mat const &m) {
  
  THTensor *tensor = THTensor_(newWithSize3d)(m.rows, m.cols, m.channels());
  
  real *data = THTensor_(data)(tensor);
  size_t size = THTensor_(nElement)(tensor);
  
  real *mat_data = (real*)(m.data);
  assert(THTensor_(isContiguous)(tensor));
  
  TH_TENSOR_APPLY(real, tensor, *tensor_data = *mat_data++; );
  
  THTensor_(transpose)(tensor, tensor, 0, 1);
  THTensor_(transpose)(tensor, tensor, 0, 2);
  
  libopencv_(CanConvert)("ToTensor", m, tensor);
  
  return tensor;
}


// 
// inline cv::Mat libopencv_(toTensor)(cv::Mat const &m, THTensor *dest) {
//   
// 
//   
// }


// inline cv::Mat_<T> libopencv_(Main_torch2cv_rgb)(THTensor *source) {


// static void libopencv_(Main_opencvPoints2torch)(CvPoint2D32f * points, int npoints, THTensor *tensor) {
// 
//   // Resize target
//   THTensor_(resize2d)(tensor, npoints, 2);
//   THTensor *tensorc = THTensor_(newContiguous)(tensor);
//   real *tensor_data = THTensor_(data)(tensorc);
// 
//   // copy
//   int p;
//   for (p=0; p<npoints; p++){
//     *tensor_data++ = (real)points[p].x;
//     *tensor_data++ = (real)points[p].y;
//   }
// 
//   // cleanup
//   THTensor_(free)(tensorc);
// }
// 
// static CvPoint2D32f * libopencv_(Main_torch2opencvPoints)(THTensor *src) {
// 
//   int count = src->size[0];
//   int dims  = src->size[1];
//   // create output
//   CvPoint2D32f * points_cv = NULL;
//   points_cv = (CvPoint2D32f*)cvAlloc(count*sizeof(points_cv[0]));
//   real * src_pt = THTensor_(data)(src);
//   // copy
//   int p,q;
//   for (p=0; p<count; p++){
//     points_cv[p].x = (float)*src_pt++ ;
//     points_cv[p].y = (float)*src_pt++ ;
//     if (dims > 2) {
//       for (q=2 ; q<dims ; q++){
//         src_pt++;
//       }
//     }
//   }
// 
//   // return freshly created CvPoint2D32f
//   return points_cv;
// }


//============================================================
static int libopencv_(Main_cvResize) (lua_State *L) {
  // Get Tensor's Info
  THTensor * source = luaT_checkudata(L, 1, torch_Tensor);
  
  int w = lua_tonumber(L, 2);
  int h = lua_tonumber(L, 3);

  int quality = lua_tonumber(L, 4);
  
  return libopencv_(WithTensor)<THTensor *>(tensor, [w, h](cv::Mat  &image) {
    
    cv::Size size(w, h);
    
    cv::Mat dest(size, image.type());
    cv::resize(image, dest, size);

    return libopencv_(ToTensor)(dest);
  }
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
//   libopencv_(WithTensor)(image, [win](cv::Mat const &image) {
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
  
  THTensor *tensor    = (THTensor*)luaT_checkudata(L, 1, torch_Tensor);
  const char* windowName     = lua_tostring(L, 2);
  windowName = windowName ? windowName : "window";
      
  libopencv_(WithTensor)<bool>(tensor, [windowName](cv::Mat  &image) {
    
    cv::startWindowThread();
    cv::namedWindow(windowName);
    
    cv::imshow(windowName, image);
    return true;
  });
    
  
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


extern "C" {

DLL_EXPORT int libopencv_(Main_init) (lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, libopencv_(Main__), "libopencv");
  lua_pop(L,1); 
  return 1;
}

}
#endif
