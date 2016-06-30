require 'nngraph'
require 'imgtools'

local cmd = torch.CmdLine()

cmd:option('-content_img','image_sets/tubingen.jpg','The image to apply style transfer on')
cmd:option('-transfer_model','models/newlytrainedmodel.t7','The model of transferring')
cmd:option('-output','output.jpg','The path to output')
cmd:option('-output_size',false,'Maximum edge of the output image')

function main(params)
	timer = torch.Timer()
	transfer_model = torch.load(params.transfer_model)

	local image = loadimg(params.content_img,params.output_size):float()
	local org_size = #image
	local content_batch_size = torch.LongStorage(4)
	content_batch_size[1] = 1
	for i=1,3 do
		content_batch_size[i+1] = (#image)[i]
	end
	image = torch.reshape(image,content_batch_size)
	local newimg = transfer_model:forward(image)
	newimg = torch.reshape(newimg,org_size)

	saveimg(newimg, params.output)
	print('Transfer complete in '.. timer:time().real ..' seconds!')
end

local params = cmd:parse(arg)
main(params)