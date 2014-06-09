#include <TH.h>
#include <luaT.h>
#include <opencv/cv.h>



static int qualityFlags(int index) {
  switch(index) {
   case 0: return CV_INTER_NN;
   case 2: return CV_INTER_CUBIC;
   case 3: return CV_INTER_AREA;
   case 4: return CV_INTER_LANCZOS4;   
   default: return CV_INTER_LINEAR;
  }
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libopencv_init [] =
{
  {NULL,NULL}
};

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor        TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libopencv_(NAME) TH_CONCAT_3(libopencv_, Real, NAME)

#include "generic/opencv.cpp"
#include "THGenerateAllTypes.h"

extern "C" {

DLL_EXPORT int luaopen_libopencv(lua_State *L)
{

  luaL_register(L, "opencv", libopencv_init);

  libopencv_ByteMain_init(L);
  libopencv_CharMain_init(L);
  libopencv_LongMain_init(L);
  libopencv_FloatMain_init(L);
  libopencv_DoubleMain_init(L);

  return 1;
}

}