require 'torch'

local affine = {}

affine.eye = function()
  return torch.eye(3):double()
end

affine.scaling = function(sx, sy) 
  sy = sy or sx

  return torch.diag(torch.DoubleTensor {sx, sy, 1})
end


affine.rotation = function(a)
  local sa = torch.sin(a)
  local ca = torch.cos(a)

  return torch.DoubleTensor { 
    {ca, -sa, 0},
    {sa,  ca, 0},
    {0,   0,  1}
  } 
end

affine.translation = function(x, y)
  return torch.DoubleTensor { 
    {1,   0,  x},
    {0,   1,  y},
    {0,   0,  1}
  } 
end


affine.around = function(x, y, t)
  return  affine.translation(x, y) * t * affine.translation(-x, -y)
end

affine.centreOn = function(source, width, height)

  local dw = (width - source:size(2)) * 0.5
  local dh = (height - source:size(1)) * 0.5
  
  local t = affine.translation(dw, dh)
  return opencv.warpAffine { image = source, width = width, height = height,  warp = t }
end

return affine