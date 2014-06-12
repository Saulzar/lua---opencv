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


local function showFlags (t)
  keys = {}
  for k, _ in pairs(t) do
    table.insert(keys, k)
  end
  
  return string.format("(%s)", table.concat(keys, ", "))
end


local function isTensor(x)
  return torch.typename(x) and torch.typename(x):find('Tensor')
end

local function swap(tensor, dim, i1, i2) 
  local x1 = tensor:narrow(dim, i1, 1)
  local x2 = tensor:narrow(dim, i2, 1)
  
  y = x1:clone()
  x1:copy(x2)
  x2:copy(y)
  
  return tensor
end


local affine = {}
opencv.affine = affine

affine.eye = torch.eye(3)

affine.scaling = function(sx, sy) 
  sy = sy or sx

  return torch.diag(torch.Tensor {sx, sy, 1})
end


affine.rotation = function(a)
  local sa = torch.sin(a)
  local ca = torch.cos(a)

  return torch.Tensor { 
    {ca, -sa, 0},
    {sa,  ca, 0},
    {0,   0,  1}
  } 
end

affine.translation = function(x, y)
  return torch.Tensor { 
    {1,   0,  x},
    {0,   1,  y},
    {0,   0,  1}
  } 
end


affine.around = function(x, y, t)
  return  affine.translation(x, y) * t * affine.translation(-x, -y)
end



-- WarpAffine
function opencv.resize(...)
  
   local flags = showFlags(libopencv.inter)
   local _, image, dest, quality, w, h = dok.unpack(
      {...},
      'opencv.Resize',
      [[Resize an image.]],
      {arg='image', type='torch.*Tensor',
       help='image to resize', req=true},
      {arg='dest', type='torch.*Tensor',
       help='destination to write the warped image', default=nil},
      {arg='quality', type='string',
       help='quality of the warp ' ..flags, default='linear'},       
      {arg='w', type='number',
       help='width of resized image', default=0},
      {arg='h', type='number',
       help='height of resized image', default=0}
   )
   
   assert(libopencv.inter[quality] ~= nil, "opencv.resize: quality must be one of "..flags)   
   assert(dest ~= nil or (h > 0 and w > 0),  "opencv.resize: either destination or (width > 0) and (height > 0) must be supplied")
     
   local dest = dest or torch.Tensor():typeAs(image):resize(h, w, image:size(3))
   image.libopencv.resize(image, dest, libopencv.inter[quality])
      
   return dest
end


-- test function:
function opencv.test.resize(img, quality)
   img = img or opencv.lena()
   
   local dest = torch.Tensor(3, 100, 150):typeAs(img)

   local scaled = opencv.Resize(img, dest, quality)
   opencv.display{image=scaled,win='Scaled image'}
end



-- WarpAffine
function opencv.warpAffine(...)
  
   local flags = showFlags(libopencv.inter)
   
   local _, image, dest, warp, quality, fill = dok.unpack(
      {...},
      'opencv.WarpAffine',
      [[Implements the affine transform which allows the user to warp,
            stretch, rotate and resize an image.]],
      {arg='image', type='torch.*Tensor',
       help='image to warp', req=true},
      {arg='dest', type='torch.Tensor',
       help='destination to write the warped image', default=nil},
      {arg='warp', type='torch.DoubleTensor',
       help='2x3 transformation matrix', req=true},
      {arg='quality', type='string',
       help='quality of the warp, one of '..flags, default='linear'},
      {arg='fill', type='boolean',
       help='fill the areas outside source image?', default=false}       
   )
     
   assert(opencv.inter[quality] ~= nil, 'opencv.WarpAffine quality must be one of '..flags)    
   assert(warp:size(1) == 2 or warp:size(1) == 3 and warp:size(2) == 3, "opencv.WarpAffine warp Tensor must be 2x3 or 3x3")
  
   local dest = dest or torch.Tensor():typeAs(image):resizeAs(image)
   image.libopencv.warpAffine(image, dest, warp:narrow(1, 1, 2):contiguous(), opencv.inter[quality], fill)
      
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



local function centreOn(source, dest)

  local ds = (torch.LongTensor(dest:size())  - torch.LongTensor(source:size())):double() * 0.5
  local t = affine.translation(ds[2], ds[1])
  
  return opencv.warpAffine { image = source, dest = dest,  warp = t }
end


function opencv.tileImageTensor(tensor, cols)
  local rows = math.ceil(tensor:size(1)/cols)
  local cols = math.min(tensor:size(1), cols)
  
  local h = tensor:size(2)
  local w = tensor:size(3)
 
  local image = torch.Tensor():typeAs(tensor):resize(rows * h, cols * w, tensor:size(4)):zero()
  
  for r = 0, rows - 1 do
    for c = 0, cols - 1 do
      local rs = r * h 
      local cs = c * w
      
      local n = r * cols + c + 1
      
      if(n <= tensor:size(1)) then
        image[{ {rs + 1, rs + h}, {cs + 1, cs + w} }] = tensor[n]
      end
    end
  end

  return image
end

function opencv.tileImageTable(images, cols)
  
  local w = 0
  local h = 0
  
  local type = torch.typename(images[1])
  local channels = images[1]:size(3)
  
  for _, img in pairs(images) do
    w = math.max(w, img:size(2))
    h = math.max(h, img:size(1))
    
    assert(torch.typename(img) == type, "opencv.display: images must have the same type")
    assert(img:size(3) == channels, "opencv.display: images must have the same number of channels")
  end
  
  local tensor = torch.Tensor():type(type):resize(#images, h, w, channels)
  for i, img in pairs(images) do
    centreOn(img, tensor[i])
  end
    
  return opencv.tileImageTensor(tensor, cols)
end





function opencv.display(...)
   local _, img, cols, win = dok.unpack({...}, 'opencv.display',
      'displays a single image, with optional parameters',
      {arg='image', type='torch.Tensor | table',
       help='image or table of images opencv BGR (HxWxCh)', req=true},   
      {arg='cols', type='number',
        help = "number of images per row", default=6},
      {arg='win', type='string', help='window legend (and descriptor)', default=nil}
   )
         
   if(isTensor(img)) then
     assert(img:size(img:dim()) <= 4, "opencv.display: image must have 1,3 or 4 channels");

     if(img:dim() == 3) then
        img.libopencv.display(img:contiguous(), win)
     elseif img:dim() == 4 then
        img.libopencv.display(opencv.tileImageTensor(img, cols), win)       
     else
       assert(false, "opencv.display: expected image (HxWxCh), or tensor of images (nxHxWxCh)")
     end
   elseif type(img) == "table" and #img > 0 then    
      img[1].libopencv.display(opencv.tileImageTable(img, cols), win)
   else
    assert(false, "opencv.display: expected image, table of images or tensor of images")
   end

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



