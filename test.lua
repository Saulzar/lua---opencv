
local affine = require 'opencv.affine'

local test = {}

-- test function:
function test.resize(img, quality)
   img = img or opencv.lena()
   
   
   local scaled = opencv.resize { image = img, height = 100, width = 150,  quality = "area" }
   opencv.display{image=scaled,win='resize'}
end



-- test function:
function test.warpAffine(img)
   img = img or opencv.lena()
   
   local warp = torch.Tensor {
     {0.2, 0.1, -15},
     {0.15, 0.3, 0}
   }
     
   local warpImg = opencv.warpAffine {image = img, height = 150, width = 150, warp = warp, quality = "linear"}  
   opencv.display{image = {warpImg}, win = 'warpAffine'}
 
end


function test.test()
  test.warpAffine()
  test.resize()
end


return test