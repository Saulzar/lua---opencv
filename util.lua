require 'torch'


local util = {}
local affine = require 'opencv.affine'


function util.isTensor(x)
  return torch.typename(x) and torch.typename(x):find('Tensor')
end

function util.swap(tensor, dim, i1, i2) 
  local x1 = tensor:narrow(dim, i1, 1)
  local x2 = tensor:narrow(dim, i2, 1)
  
  y = x1:clone()
  x1:copy(x2)
  x2:copy(y)
  
  return tensor
end



function util.tileImageTensor(tensor, cols)
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

function util.tileImageTable(images, cols)
  
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
    tensor[i] = affine.centreOn(img, w, h)
  end
    
  return util.tileImageTensor(tensor, cols)
end


function util.smoothWarp(args) 

  local warp = affine.eye()
  warp:narrow(1, 1, 2):copy(args.warp:narrow(1, 1, 2))
  
  local factor = args.factor or 2
  
  args.warp = warp * affine.scaling(factor)
  args.width = args.width * factor
  args.height = args.height * factor
  
  local image = opencv.warpAffine(args)
  return opencv.resize { image = image, scale = 1.0 / factor, quality = "area" }
  
end



return util