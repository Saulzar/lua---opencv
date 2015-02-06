
require 'torch'
require 'sys'
require 'paths'
require 'dok'


-- load C lib
require 'libopencv'

opencv = { }

local util = require 'opencv.util'
local affine = require 'opencv.affine'

opencv.util = util
opencv.affine = affine


local function showFlags (t)
  keys = {}
  for k, _ in pairs(t) do
    table.insert(keys, k)
  end
  
  return string.format("(%s)", table.concat(keys, ", "))
end


-- WarpAffine
function opencv.resize(...)
  
   local flags = showFlags(libopencv.inter)
   local _, image, width, height, quality, scale = dok.unpack(
      {...},
      'opencv.resize',
      [[Resize an image.]],
      {arg='image', type='torch.*Tensor',
       help='image to resize', req=true},
  
      {arg='width', type='number',
       help='width of resized image', default = 0},
      {arg='height', type='number',
       help='height of resized image', default = 0},
 
     {arg='quality', type='string',
       help='quality of the resize ' ..flags, default='linear'},           
        
      {arg='scale', type='number',
       help='use scale factor of input image', default=1}
   )
   
   assert(libopencv.inter[quality] ~= nil, "opencv.resize: quality must be one of "..flags)   
   
   if(width == 0) then
    width = image:size(2) * scale
   end
   
   if(height == 0) then
    height = image:size(1) * scale
   end   
   
   assert(image:isContiguous(), "opencv.resize: image must be contiguous")   
   return image.libopencv.resize(image, width, height, libopencv.inter[quality])
end



-- WarpAffine
function opencv.batchResize(...)
  
   local flags = showFlags(libopencv.inter)
   local _, images, width, height, quality, scale = dok.unpack(
      {...},
      'opencv.resize',
      [[Resize an image.]],
      {arg='images', type='torch.*Tensor',
       help='images to resize', req=true},
   
       
      {arg='width', type='number',       
       help='width of resized image', default=0},
      {arg='height', type='number',
       help='height of resized image', default=0},
      {arg='quality', type='string',
       help='quality of the resize ' ..flags, default='linear'},
      {arg='scale', type='number',
       help='height of resized image', default=1}
   )

   assert(libopencv.inter[quality] ~= nil, "opencv.batchResize: quality must be one of "..flags)   
   assert((scale ~= 1) or (height > 0 and width > 0),  "opencv.batchResize: either supply scale or (width, height)")
        
   if(util.isTensor(images)) then
    
    if(scale ~= 1) then
      width = images:size(3) * scale
      height = images:size(2) * scale
    end           
     
    assert(images:isContiguous(), "opencv.batchResize: image must be contiguous")        
    assert(images:dim() == 4, "opencv.batchResize: expected tensor of nxHxWxCh")   
    
    local dest = dest or torch.Tensor():typeAs(images):resize(images:size(1), height, width, images:size(4))
    for i = 1, images:size(1) do
      dest[i]:copy(images.libopencv.resize(images[i], width, height, libopencv.inter[quality]))      
    end
    
    return dest
     
   elseif type(images) == "table" and #images > 0 then    
    local result = {}
    for i, image in pairs(images) do
     
      local w = width
      local h = height
      
      if(scale ~= 1) then
        w = image:size(2) * scale
        h = image:size(1) * scale
      end      
        
      result[i] = image.libopencv.resize(image, w, h, libopencv.inter[quality])   
    end
    
    return result   
   else
    assert(false, "opencv.display: table of images or tensor of images")
   end
end


-- WarpAffine
function opencv.warpAffine(...)
  
   local flags = showFlags(libopencv.inter)
   
   local _, image, width, height, warp, quality, fill = dok.unpack(
      {...},
      'opencv.warpAffine',
      [[Implements the affine transform which allows the user to warp,
            stretch, rotate and resize an image.]],
      {arg='image', type='torch.*Tensor',
       help='image to warp', req=true},
      
      {arg='width', type='number',
       help='width of result', req = true},
      {arg='height', type='number',
       help='height of result', req = true},
       
      {arg='warp', type='torch.DoubleTensor',
       help='2x3 transformation matrix', req=true},
      {arg='quality', type='string',
       help='quality of the warp, one of '..flags, default='linear'},
      {arg='fill', type='boolean',
       help='fill the areas outside source image?', default=false}       
   )
   
  
   assert(libopencv.inter[quality] ~= nil, 'opencv.warpAffine quality must be one of '..flags)    
   assert(warp:size(1) == 2 or warp:size(1) == 3 and warp:size(2) == 3, "opencv.warpAffine warp Tensor must be 2x3 or 3x3")
  
   assert(util.isTensor(image), "opencv.warpAffine : image must be a tensor")  
   
   width = width or image:size(2)
   height = height or image:size(1)
   
   return image.libopencv.warpAffine(image, warp:narrow(1, 1, 2):contiguous():double(), width, height, libopencv.inter[quality], fill)
end

function opencv.lena()
  return opencv.load(paths.concat(sys.fpath(), 'lena.jpg'))
end




function opencv.display(...)
    
   local _, image, cols, win, scale = dok.unpack({...}, 'opencv.display',
      'displays a single image, with optional parameters',
      {arg='image', type='torch.Tensor | table',
       help='image or table of images opencv BGR (HxWxCh)', req=true},   
      {arg='cols', type='number',
        help = "number of images per row", default=6},
      {arg='win', type='string', help='window legend (and descriptor)', default=nil},
      {arg='scale', type='number',
        help = "scale images for display", default=1}
   )
   
   local function display(image)  
      if (scale ~= 1) then
        image = opencv.resize {image = image, scale = scale, quality = "area"}
      end
      image.libopencv.display(image:contiguous(), win)
   end
   
         
   if(util.isTensor(image)) then
     assert(image:size(image:dim()) <= 4, "opencv.display: image must have 1,3 or 4 channels");

     if(image:dim() == 3) then
        display(image)
     elseif image:dim() == 4 then
       display(util.tileImageTensor(image, cols))
     else
       assert(false, "opencv.display: expected image (HxWxCh), or tensor of images (nxHxWxCh)")
     end
   elseif type(image) == "table" and #image > 0 then    
      display(util.tileImageTable(image, cols))     
   else
    assert(false, "opencv.display: expected image, table of images or tensor of images")
   end

end


function opencv.load(...)
   local _, filename = dok.unpack({...}, 'opencv.load',
      'loads an image from disk',
      {arg='filename', type='string',
       help='image file to load', req=true}
   )
      
   return libopencv.load(filename)
end


function opencv.save(...)
   local _, filename, image = dok.unpack({...}, 'opencv.save',
      'save an image to disk',
      {arg='filename', type='string',
       help='filename of ', req=true},
      {arg='image', type='ByteTensor | ShortTensor',
       help='image of 1, 3, or 4 channels HxWxCh', req=true}   
   )
      
   return image.libopencv.save(filename, image)
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
   local _, image, code = dok.unpack({...}, "opencv.convertColor", 
      'convert an image from one colour format to another',
      {arg='image', type='torch.FloatTensor | torch.ByteTensor | torch.ShortTensor',
       help= 'image of (HxWxc)', req=true},
      {arg='code', type='string',
       help='requested conversion, one of: '..codes, req = true}
   )
  
  assert(libopencv.convert[code], "opencv.convertColor: conversion must be one of "..codes)
  return image.libopencv.convertColor(image, libopencv.convert[code])
end



function opencv.test()
  local test = require "opencv.test"
  test.test()
  
end

return opencv