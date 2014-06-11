--
-- Note: this bit of code is a simple wrapper around the OpenCV library
--       http://opencv.willowgarage.com/
--
-- For now, it contains wrappers for:
--  + opencv.GetAffineTransform() [lua]    --> cvGetAffineTransform [C/C++]
--  + opencv.WarpAffine() [lua]            --> cvWarpAffine [C/C++]
--  + opencv.EqualizeHist() [lua]          --> cvEqualizeHist [C/C++]
--  + opencv.Canny() [lua]                 --> cvCanny [C/C++]
--  + opencv.CornerHarris() [lua] --> cvCornerHarris [C/C++]
--
--  + opencv.CalcOpticalFlow() [lua] -->
--    - cvCalcOpticalFlowBM
--    - cvCalcOpticalFlowHS
--    - cvCalcOpticalFlowLK
--
--  + opencv.GoodFeaturesToTrack() [lua] --> cvGoodFeaturesToTrack [C/C++]
--
-- Wrapper: Clement Farabet.
-- Additional functions: GoodFeatures...(),PLK,etc.: Marco Scoffier
-- Adapted for torch7: Marco Scoffier
--

require 'torch'
require 'sys'
require 'paths'
require 'dok'



-- load C lib
require 'libopencv'

opencv = { test = {}, convert = libopencv.convert, inter = libopencv.inter }

-- WarpAffine
function opencv.resize(...)
  
   local flags = showFlags(libopencv.inter)
   local _, source, dest, quality, w, h = dok.unpack(
      {...},
      'opencv.Resize',
      [[Resize an image.]],
      {arg='source', type='torch.Tensor | torch.ByteTensor',
       help='source image to warp', req=true},
      {arg='dest', type='torch.Tensor',
       help='destination to write the warped image', default=nil},
      {arg='quality', type='string',
       help='quality of the warp ' ..flags, default='linear'},       
      {arg='w', type='number',
       help='width of resized image', default=0},
      {arg='h', type='number',
       help='height of resized image', default=0}
   )
   
   if(libopencv.inter[quality] == nil) then
     xerror(' *** ERROR: opencv.Resize quality must be one of ' ..flags)
   end
   
   if(dest == nil and (h == 0 or w == 0)) then
     xerror(' *** ERROR: opencv.Resize, either destination or (width > 0) and (height > 0) must be supplied')
   end
     
   local dest = dest or torch.Tensor():typeAs(source):resize(h, w, source:size(3))
   source.libopencv.resize(source, dest, qualityTypes[quality])
      
   return dest
end


-- test function:
function opencv.test.resize(img, quality)
   img = img or opencv.lena()
   
   local dest = torch.Tensor(3, 100, 150):typeAs(img)

   local scaled = opencv.Resize(img, dest, quality)
   opencv.display{image=scaled,win='Scaled image'}
end


local function showFlags (t)
  keys = {}
  for k, _ in pairs(t) do
    table.insert(keys, k)
  end
  
  return string.format("(%s)", table.concat(keys, ", "))
end

-- WarpAffine
function opencv.warpAffine(...)
  
   local flags = showFlags(libopencv.inter)
   
   local _, source, dest, warp, quality, fill = dok.unpack(
      {...},
      'opencv.WarpAffine',
      [[Implements the affine transform which allows the user to warp,
            stretch, rotate and resize an image.]],
      {arg='source', type='torch.*Tensor',
       help='source image to warp', req=true},
      {arg='dest', type='torch.Tensor',
       help='destination to write the warped image', default=nil},
      {arg='warp', type='torch.DoubleTensor',
       help='2x3 transformation matrix', req=true},
      {arg='quality', type='string',
       help='quality of the warp, one of '..flags, default='linear'},
      {arg='fill', type='boolean',
       help='fill the areas outside source image?', default=false}       
   )
     
   if(opencv.inter[quality] == nil) then
     xerror(' *** ERROR: opencv.WarpAffine quality must be one of '..flags)
   end
   
   
   if warp:size(1) ~= 2 or warp:size(2) ~= 3 then
      xerror(' *** ERROR: opencv.WarpAffine warp Tensor must be 2x3')
   end
   
  
   local dest = dest or torch.Tensor():typeAs(source):resizeAs(source)
   source.libopencv.warpAffine(source, dest, warp, opencv.inter[quality], fill)
      
   return dest
end

function opencv.lena()
  return opencv.load(paths.concat(sys.fpath(), 'lena.jpg'))
end

-- test function:
function opencv.test.warpAffine(img, quality, fill)
   img = img or opencv.lena()
   
   local warp = torch.Tensor {
     {0.4,   0.2, 0},
     {0.3, 0.4, 0}
   }
   
   local warpImg = opencv.warpAffine(img, nil, warp, quality, fill)  
   opencv.display{image=warpImg, win='Warped image'}
   
end


function swap(tensor, dim, i1, i2) 
  local x1 = tensor:narrow(dim, i1, 1)
  local x2 = tensor:narrow(dim, i2, 1)
  
  y = x1:clone()
  x1:copy(x2)
  x2:copy(y)
  
  return tensor
end
  
  

function opencv.display(...)
   local _, img, win = dok.unpack({...}, 'opencv.display',
      'displays a single image, with optional parameters',
      {arg='image', type='torch.Tensor | table',
       help='image or table of images opencv BGR (HxWxCh)', req=true},   
      {arg='win', type='string', help='window legend (and descriptor)', default=nil}
   )
   
   assert(img:size(3) <= 4, "opencv.display: image must have 1,3 or 4 channels");
   
   img.libopencv.display(img:contiguous(), win)
end


function opencv.load(...)
   local _, filename = dok.unpack({...}, 'opencv.display',
      'loads an image from disk',
      {arg='filename', type='string',
       help='image or table of images 3xHxW', req=true}
   )
      
   return libopencv.load(filename)
end

function opencv.toCV(...)
   local _, image = dok.unpack({...}, 'opencv.display',
      'transpose an image from torch image RGB (ChxHxW) to opencv BGR (HxWxCh)',
      {arg='image', type='torch.Tensor',
       help='image', req=true}
   )
  
  local cv = image:clone()
  cv = swap(cv, 1, 1, 3)
  cv = cv:transpose(1, 3)
  cv = cv:transpose(1, 2)
  
  return cv:contiguous()
end


function opencv.convertColor(...)
   local codes = showFlags(libopencv.convert)
   local _, image, code, dest = dok.unpack({...}, "opencv.convertColor", 
      'convert an image from one colour format to another',
      {arg='image', type='torch.FloatTensor | torch.ByteTensor | torch.ShortTensor',
       help= 'image of (HxWxc)', req=true},
      {arg='code', type='string',
       help='requested conversion, one of: '..codes, req = true}
   )
  
  assert(libopencv.convert[code] ~= nil, "opencv.convertColor: conversion must be one of "..codes)
  return image.libopencv.convertColor(image, libopencv.convert[code])
end



