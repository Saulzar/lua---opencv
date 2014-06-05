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
require 'dok'
require 'image'

opencv = {}

-- load C lib
require 'libopencv'



function opencv.GetAffineTransform(...)
   local args,  points_src, points_dst  = dok.unpack(
      {...},
      'opencv.GetAffineTransform',
      [[Calculates the affine transform from 3 corresponding points. ]],
      {arg='points_src',type='torch.Tensor',
       help='source points', req=true},
      {arg='points_dst',type='torch.Tensor',
       help='destination points', req=true}
   )
   local warp = torch.Tensor()
   warp.libopencv.GetAffineTransform(points_src,points_dst,warp)
   
   return warp[1]
end

-- test function:
function opencv.GetAffineTransform_testme()
   src = torch.Tensor(3,2)
   dst = torch.Tensor(3,2)
   src[1][1]=0
   src[1][2]=0
   src[2][1]=511
   src[2][2]=0
   src[3][1]=0
   src[3][2]=511

   dst[1][1]=0
   dst[1][2]=512*0.25
   dst[2][1]=512*0.9
   dst[2][2]=512*0.15
   dst[3][1]=512*0.1
   dst[3][2]=512*0.75

   warp = opencv.GetAffineTransform(src,dst)
   print('Warp matrix:')
   print(warp)
end

local qualityTypes = {
  nn = 0,
  linear  = 1,
  cubic   = 2,
  area    = 3,
  lanczos  = 4
}

-- WarpAffine
function opencv.Resize(...)
   local _, source, dest, quality, w, h = dok.unpack(
      {...},
      'opencv.Resize',
      [[Resize an image.]],
      {arg='source', type='torch.Tensor | torch.ByteTensor',
       help='source image to warp', req=true},
      {arg='dest', type='torch.Tensor',
       help='destination to write the warped image', default=nil},
      {arg='quality', type='string',
       help='quality of the warp (nn, linear, cubic, area, lanczos)', default='linear'},       
      {arg='w', type='number',
       help='width of resized image', default=0},
      {arg='h', type='number',
       help='height of resized image', default=0}
   )
   
   if(qualityTypes[quality] == nil) then
     xerror(' *** ERROR: opencv.Resize quality must be one of (nn, linear, cubic, area, lanczos)')
   end
   
   if(dest == nil and (h == 0 or w == 0)) then
     xerror(' *** ERROR: opencv.Resize, either destination or (width > 0) and (height > 0) must be supplied')
   end
     
   local dest = dest or torch.Tensor():typeAs(source):resize(source:size(1), h, w)
   source.libopencv.Resize(source, dest, qualityTypes[quality])
      
   return dest
end


-- test function:
function opencv.Resize_testme(img, quality)
   img = img or image.lena()
   
   local dest = torch.Tensor(3, 100, 150):typeAs(img)

   local scaled = opencv.Resize(img, dest, quality)
   opencv.display{image=scaled,win='Scaled image'}
end


-- WarpAffine
function opencv.WarpAffine(...)
   local _, source, dest, warp, quality, fill = dok.unpack(
      {...},
      'opencv.WarpAffine',
      [[Implements the affine transform which allows the user to warp,
            stretch, rotate and resize an image.]],
      {arg='source', type='torch.*Tensor',
       help='source image to warp', req=true},
      {arg='dest', type='torch.Tensor',
       help='destination to write the warped image', default=nil},
      {arg='warp', type='torch.Tensor',
       help='2x3 transformation matrix', req=true},
      {arg='quality', type='string',
       help='quality of the warp (nn, linear, cubic, area)', default='linear'},
      {arg='fill', type='boolean',
       help='fill the areas outside source image?', default=false}       
   )
     
   if(qualityTypes[quality] == nil) then
     xerror(' *** ERROR: opencv.WarpAffine quality must be one of (nn, linear, cubic, area)')
   end
   
   print(warp)
   
   if warp:size(1) ~= 2 or warp:size(2) ~= 3 then
      xerror(' *** ERROR: opencv.WarpAffine warp Tensor must be 2x3')
   end
   
  
   local dest = dest or torch.Tensor():typeAs(source):resizeAs(source)
   source.libopencv.WarpAffine(source, dest, warp, qualityTypes[quality], fill)
      
   return dest
end

-- test function:
function opencv.WarpAffine_testme(img, quality, fill)
   img = img or image.lena()

   local src = torch.Tensor(3,2)
   local dst = torch.Tensor(3,2)
   src[1][1]=0
   src[1][2]=0
   src[2][1]=511
   src[2][2]=0
   src[3][1]=0
   src[3][2]=511

   dst[1][1]=0
   dst[1][2]=512*0.15
   dst[2][1]=512*0.9
   dst[2][2]=512*0.05
   dst[3][1]=512*0.1
   dst[3][2]=512*0.75

   local warp = opencv.GetAffineTransform(src,dst)
   print('warp',warp)
   local warpImg = opencv.WarpAffine(img, nil, warp, quality, fill)
   opencv.display{image=warpImg,win='Warped image'}
end



function opencv.display(...)
   local image, win = dok.unpack({...}, 'opencv.display',
      'displays a single image, with optional parameters',
      {arg='image', type='torch.Tensor',
       help='image or table of images 3xHxW', req=true},
      {arg='win', type='string', help='window legend (and descriptor)', default=nil}
   )
   
   image.libopencv.display(image, win)
end
