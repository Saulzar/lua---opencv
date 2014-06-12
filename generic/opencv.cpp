#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/opencv.cpp"
#else


#include <luaT.h>
#include <TH.h>


#include <opencv2/opencv.hpp>

inline void libopencv_(push)(lua_State *L, THTensor *tensor) {
  THTensor_(retain)(tensor);
  luaT_pushudata(L, tensor, torch_Tensor);
}


inline THTensor *libopencv_(checkTensor)(lua_State* L, int arg) {
  return (THTensor*)luaT_checkudata(L, arg, torch_Tensor);  
}

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
      
  cvAssert (false, "cvDepth: unknown type");
}



inline cv::Mat libopencv_FloatToMat(THFloatTensor *tensor);
inline cv::Mat libopencv_DoubleToMat(THDoubleTensor *tensor);

inline cv::Mat libopencv_(ToMat)(THTensor *tensor) {
  
  int channels = tensor->nDimension >= 3 ? tensor->size[2] : 1;
  int depth = libopencv_(cvDepth)();
  
  real *data = THTensor_(data)(tensor);   
  cvAssert (THTensor_(isContiguous)(tensor), "ToMat: tensor must be contiguous");
  
  return cv::Mat(tensor->size[0], tensor->size[1], CV_MAKETYPE(depth, channels), static_cast<void*>(data));
}




inline THTensor *libopencv_(ToTensor)(cv::Mat const &m) {
  THTensor *tensor = THTensor_(newWithSize3d)(m.rows, m.cols, m.channels());
  
  real *data = THTensor_(data)(tensor);
  size_t size = THTensor_(nElement)(tensor);
  
  real *mat_data = (real*)(m.data);
  cvAssert (THTensor_(isContiguous)(tensor), "ToTensor: tensor should be contiguous");
  
  TH_TENSOR_APPLY(real, tensor, *tensor_data = *mat_data++; );
  return tensor;
}




//============================================================
static int libopencv_(cvResize) (lua_State *L) {
  
  try {
    // Get Tensor's Info
    THTensor * source_tensor = libopencv_(checkTensor)(L, 1);
    THTensor * dest_tensor = libopencv_(checkTensor)(L, 2);
    
    int quality = lua_tonumber(L, 3);
    
    cv::Mat source = libopencv_(ToMat)(source_tensor);
    cv::Mat dest = libopencv_(ToMat)(dest_tensor);
      
    cv::resize(source, dest, dest.size());

    libopencv_(push)(L, dest_tensor);
    return 1;    
    
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }    
  
}




//============================================================
static int libopencv_(cvWarpAffine) (lua_State *L) {

  try {
    
    THTensor * source_tensor = libopencv_(checkTensor)(L, 1);
    THTensor * dest_tensor   = libopencv_(checkTensor)(L, 2);
    THDoubleTensor * warp_tensor   = (THDoubleTensor *)luaT_checkudata(L, 3, "torch.DoubleTensor");

    int quality = lua_tonumber(L, 4);
    int fill = lua_toboolean(L, 5);

    cvAssert(warp_tensor->size[0] == 2 , "warp matrix: 2x3 Tensor expected");
    cvAssert(warp_tensor->size[1] == 3 , "warp matrix: 2x3 Tensor expected");
   
    cv::Mat warp = libopencv_DoubleToMat(warp_tensor);
    cv::Mat source = libopencv_(ToMat)(source_tensor);
    cv::Mat dest = libopencv_(ToMat)(dest_tensor);
    
    int flags = (fill ? CV_WARP_FILL_OUTLIERS : 0) | quality;  
    cv::warpAffine(source, dest, warp, cv::Size(), flags);
    
    libopencv_(push)(L, dest_tensor);
    return 1;    
    
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }
}


//============================================================
static int libopencv_(cvConvertColor) (lua_State *L) {

  try {
    
    THTensor * source_tensor = libopencv_(checkTensor)(L, 1);
    int code = lua_tonumber(L, 2);
    
    cv::Mat source = libopencv_(ToMat)(source_tensor);
    cv::Mat dest;
      
    cv::cvtColor(source, dest, code);   
    
    luaT_pushudata(L, libopencv_(ToTensor)(dest), torch_Tensor);
    return 1;    
    
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }
}



static int libopencv_(clampC) (lua_State *L) {
  try {
  
    THTensor *tensor = libopencv_(checkTensor)(L, 1);
    
    real min = lua_tonumber(L, 2);
    real max = lua_tonumber(L, 3);

    TH_TENSOR_APPLY(real, tensor, *tensor_data = clamp(*tensor_data, min, max); );
    
    libopencv_(push)(L, tensor);
    return 1;    
    
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }    
}





static int libopencv_(wrapC) (lua_State *L) {
  try {
  
    THTensor *tensor = libopencv_(checkTensor)(L, 1);
    
    real r = lua_tonumber(L, 2);
    TH_TENSOR_APPLY(real, tensor, *tensor_data = wrap(*tensor_data, r); );
    
    libopencv_(push)(L, tensor);
    return 1;       
    
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }    
}



static int libopencv_(cvDisplay)(lua_State* L) {
  try {
    THTensor *tensor    = libopencv_(checkTensor)(L, 1);
    const char* windowName     = lua_tostring(L, 2);
    windowName = windowName ? windowName : "window";

    cv::Mat image = libopencv_(ToMat)(tensor);
      
    cv::startWindowThread();
    cv::namedWindow(windowName);
    
    cv::imshow(windowName, image);  
  } catch (std::exception const &e) {
    luaL_error(L, e.what());
  }
  
  return 0;
}





//============================================================
// Register functions in LUA
//
static const luaL_reg libopencv_(Main__) [] =
{
  {"warpAffine",           libopencv_(cvWarpAffine)},
  {"resize",               libopencv_(cvResize)},
  {"display",              libopencv_(cvDisplay)},
  {"wrapC",                libopencv_(wrapC)},
  {"clampC",               libopencv_(clampC)},
  {"convertColor",         libopencv_(cvConvertColor)},
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
