#include <TH.h>
#include <luaT.h>
#include <opencv/cv.h>






inline void cvAssert (bool condition, const char *message) {
 
  if(!condition)
    throw std::invalid_argument(message);  
}


template<typename T>
inline T clamp(T const& v, T const &min, T const &max) {
  const T t = v < min ? min : v;
  return t > max ? max : t;
}

template<typename T>
inline T wrap(T const& v, T const &r) {
  auto d = std::div(v, r);
  return v > 0 ? d.rem : d.rem + r;
}

template<>
inline double wrap<double>(double const& v, double const &r) {  
  return v > 0 ? std::fmod(v, r) : std::fmod(v, r) + r;
}

template<>
inline float wrap<float>(float const& v, float const &r) {  
  return v > 0 ? std::fmod(v, r) : std::fmod(v, r) + r;
}

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor        TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libopencv_(NAME) TH_CONCAT_3(libopencv_, Real, NAME)

#include "generic/opencv.cpp"
#include "THGenerateAllTypes.h"

  


inline int pushAsTensor(lua_State* L, cv::Mat const &m) {
  
  switch(m.depth()) {
    case CV_8U:   
      luaT_pushudata(L, libopencv_ByteToTensor(m), "torch.ByteTensor");
    break;
    case CV_8S:   
      luaT_pushudata(L, libopencv_CharToTensor(m), "torch.CharTensor");
    break;
    case CV_16U:  
      luaT_pushudata(L, libopencv_ShortToTensor(m), "torch.ShortTensor");
    break;
    case CV_16S:  
      luaT_pushudata(L, libopencv_ShortToTensor(m), "torch.ShortTensor");    
    break;
    case CV_32S:  
      luaT_pushudata(L, libopencv_LongToTensor(m), "torch.LongTensor");
    break;
    case CV_32F:  
      luaT_pushudata(L, libopencv_FloatToTensor(m), "torch.FloatTensor");
    break;
    case CV_64F:  
      luaT_pushudata(L, libopencv_DoubleToTensor(m), "torch.DoubleTensor");
    break;
  }
  
  return 1;
}


std::map<std::string, int> interFlags = {
  {"nn", CV_INTER_NN}, 
  {"linear", CV_INTER_LINEAR},  
  {"cubic", CV_INTER_CUBIC}, 
  {"area", CV_INTER_AREA},
  {"lanczos4", CV_INTER_LANCZOS4}
};


std::map<std::string, int> convertFlags = {
  {"bgr2bgra", cv::COLOR_BGR2BGRA},
  {"bgr2bgra", cv::COLOR_BGRA2BGR},
  
  {"bgr2rgb", cv::COLOR_BGR2RGB},
  {"rgb2bgr", cv::COLOR_RGB2BGR},
  
  {"bgr2gray", cv::COLOR_BGR2GRAY},
  {"gray2bgr", cv::COLOR_GRAY2BGR},
  
  {"bgr2xyz", cv::COLOR_BGR2XYZ},
  {"xyz2bgr", cv::COLOR_XYZ2BGR},
  
  {"bgr2hsv", cv::COLOR_BGR2HSV},
  {"hsv2bgr", cv::COLOR_HSV2BGR},
  
  {"bgr2hls", cv::COLOR_BGR2HLS},
  {"hls2bgr", cv::COLOR_HLS2BGR},
  
  {"bgr2lab", cv::COLOR_BGR2Lab},
  {"lab2bgr", cv::COLOR_Lab2BGR}
};
    


inline void pushValue(lua_State* L, std::string const &s) {
  lua_pushstring(L, s.c_str());
}

inline void pushValue(lua_State* L, int v) {
  lua_pushnumber(L, v);
}

template<typename K, typename V>
inline void pushValue(lua_State* L, std::map<K, V> const &t) {
  lua_newtable(L);
  int top = lua_gettop(L);

  for (auto it = t.begin(); it != t.end(); ++it) {

    pushValue(L, it->first);
    pushValue(L, it->second);
    lua_settable(L, top);
  } 
  
}

template<typename K, typename V>
void setField(lua_State* L, std::string const &table, K const &key,  V const &value) {
  lua_getglobal(L, table.c_str());
  pushValue(L, key);
  pushValue(L, value);
  lua_settable(L, -3);
  lua_pop(L, 1);  
}


static int loadImage(lua_State* L) {
  const char* fileName     = lua_tostring(L, 1);
  
  cv::Mat m = cv::imread(fileName, CV_LOAD_IMAGE_UNCHANGED); 
  pushAsTensor(L, m);

  return 1;
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libopencv_init [] =
{  
  {"load",   loadImage},
  {NULL,NULL}
};


extern "C" {

DLL_EXPORT int luaopen_libopencv(lua_State *L)
{

  luaL_register(L, "libopencv", libopencv_init);

  libopencv_ByteMain_init(L);
  libopencv_CharMain_init(L);
  libopencv_LongMain_init(L);
  libopencv_FloatMain_init(L);
  libopencv_DoubleMain_init(L);
  
  setField(L, "libopencv", "inter", interFlags);
  setField(L, "libopencv", "convert", convertFlags);


  return 1;
}

}