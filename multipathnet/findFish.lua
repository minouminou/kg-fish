--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Run MultiPathNet on data')
cmd:text()
cmd:text('Options:')
cmd:option('-np', 5,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')
cmd:option('-img','./deepmask/data/testImage.jpg' ,'path/to/test/image')
cmd:option('-thr', 0.5, 'multipathnet score threshold [0,1]')
cmd:option('-maxsize', 600, 'resize image dimension')
cmd:option('-sharpmask_path', 'data/models/sharpmask.t7', 'path to sharpmask/deepmask')
cmd:option('-multipath_path', 'data/models/resnet18_integral_coco.t7', 'path to multipathnet')
cmd:option('-imgdir','data/' ,'path/to/test/dir')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:text('Save Results:')
cmd:option('-resfile', '', 'result file')
cmd:option('-saveimg', false, 'save result images')
cmd:option('-savedir','res/' ,'path/to/save/dir')

local config = cmd:parse(arg)

manualSeed = config.seed
if manualSeed == -1 then --random
   manualSeed = torch.random(10000)
end
cutorch.setDevice(config.gpu)
math.randomseed(manualSeed)
cutorch.manualSeedAll(manualSeed)
torch.manualSeed(manualSeed)

if config.dm then
  require 'deepmask.DeepMask'
  require 'deepmask.InferDeepMask'
else
  require 'deepmask.SharpMask'
  require 'deepmask.SpatialSymmetricPadding'
  require 'deepmask.InferSharpMask'
end
require 'inn'
require 'fbcoco'
require 'image'
local model_utils = require 'models.model_utils'
local utils = require 'utils'
local coco = require 'coco'

local resfile = torch.DiskFile(config.resfile, 'w')
resfile:writeString('file,file_w,file_h,prob,class,name, x1,y1,x3,y2\n')
resfile:synchronize()

local sharpmask = torch.load(config.sharpmask_path).model
sharpmask:inference(config.np)

local multipathnet = torch.load(config.multipath_path)
multipathnet:evaluate()
multipathnet:cuda()
model_utils.testModel(multipathnet)

local detector = fbcoco.ImageDetect(multipathnet, model_utils.ImagenetTransformer())

------------------- Run DeepMask --------------------

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end
print(scales)

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = sharpmask,
  dm = config.dm,
}

f = io.popen('ls ' .. config.imgdir)
for file in f:lines() do 
  if file ~= "." and file ~= ".." then
    print("Found file: " .. file)

local img = image.load(config.imgdir .. '/' .. file)
img = image.scale(img, config.maxsize)
local h,w = img:size(2),img:size(3)

infer:forward(img)

local masks,_ = infer:getTopProps(.2,h,w)

local Rs = coco.MaskApi.encode(masks)
local bboxes = coco.MaskApi.toBbox(Rs)
bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2

------------------- Run MultiPathNet --------------------

local detections = detector:detect(img:float(), bboxes:float())
local prob, maxes = detections:max(2)

-- remove background detections
--local idx = maxes:squeeze():gt(1):cmul(prob:gt(config.thr)):nonzero():select(2,1) --
--bboxes = bboxes:index(1, idx)
--maxes = maxes:index(1, idx)
--prob = prob:index(1, idx)

local scored_boxes = torch.cat(bboxes:float(), prob:float(), 2)
local final_idx = utils.nms_dense(scored_boxes, 0.3)

------------------- Draw detections --------------------

-- remove suppressed masks
--masks = masks:index(1, idx):index(1, final_idx)

local dataset = paths.dofile'./DataSetJSON.lua':create'coco_val2014'

local res = img:clone()
coco.MaskApi.drawMasks(res, masks, 10)
for i,v in ipairs(final_idx:totable()) do
     local class = maxes[v][1]-1
     local x1,y1,x2,y2 = table.unpack(bboxes[v]:totable())
     y2 = math.min(y2, res:size(2)) - 10
     local name = dataset.dataset.categories[class]
     print(prob[v][1], class, name)
     if config.saveimg then image.drawText(res, name, x1, y2, {bg={255,255,255}, inplace=true}) end
     --print(file, w, h, prob[v][1], class, name, x1,y1,x2,y2)
     local logstr = string.format('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',file, w, h, prob[v][1], class, name, x1,y1,x2,y2)
     resfile:writeString(logstr)
     resfile:synchronize()
end
if config.saveimg then image.save(string.format('%s/%s',config.savedir,file),res) end

  end 
end
print('| done')
