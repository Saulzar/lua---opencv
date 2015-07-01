
local affine = require 'opencv.affine'

local test = {}
-- local opencv = require 'opencv'

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


function test.inpaint(img, quality)
   img = img or opencv.lena()
   img = opencv.convertColor(img, "bgr2gray")

   
   local scaled = opencv.resize { image = img, height = 200, width = 300,  quality = "area" }
   
   local size = scaled:size()
   size[3] = 1
   
   mask = torch.Tensor(size):uniform(0, 1):gt(0.6)
   inv = mask:clone():mul(-1):add(1)
   
   scaled = scaled:cmul(inv:typeAs(scaled))
   
   local inpainted = opencv.inpaint {image = scaled:clone(), mask = mask, radius = 8}
--    local inpainted = opencv.medianBlur {image = scaled, kernel = 7}
     
   opencv.display{image=mask:mul(255),win='mask'}   
   opencv.display{image=scaled,win='input'}
   opencv.display{image=inpainted,win='inpaint'}

end


function test.distanceTransform(img, quality)

  local mask = torch.ByteTensor (400, 400, 1):fill(255)
  local rect = mask:narrow(2, 170, 60):narrow(1, 170, 60)
  rect:fill(0)
 
  opencv.display{image=mask,win='mask'}   

  local d = opencv.distanceTransform(mask)
  print {d}
  
  opencv.display{image=d:mul(2/255),win='distances'}   
  
end



function test.test()
--   test.warpAffine()
--   test.resize()
  test.distanceTransform()
  
  io.read()
end

-- test.test()


return test