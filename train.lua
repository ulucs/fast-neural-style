require 'networks'
require 'imgtools'
require 'optim'

local cmd = torch.CmdLine()

cmd:option('-style_img', 'style_images/starry.jpg', 'Style image to train on the image set')
-- next option is going to be the training set option when I get around
cmd:option('-training_img', 'image_sets/tubingen.jpg', 'Placeholder single training image')
cmd:option('-modelfile', 'models/VGG_ILSVRC_16_layers.caffemodel', 'Model used for perceptual losses')
cmd:option('-protofile', 'models/VGG_ILSVRC_16_layers_deploy.prototxt', 'prototxt of the perception model')
cmd:option('-style_weight', 1e2)
cmd:option('-content_weight', 5e0)
cmd:option('-tv_weight', 1e-3)
cmd:option('-style_layers_t', 'relu1_2,relu2_2,relu3_3,relu4_3')
cmd:option('-content_layers_t', 'relu2_2')
cmd:option('-iterations', 100, 'Number of iterations to run for training')
cmd:option('-trained_model', 'models/newlytrainedmodel.t7', '')

local function main(params)

	-- Load the content and style images
	local content_img = loadimg(params.training_img,256):float()
	local content_batch_size = torch.LongStorage(4)
	content_batch_size[1] = 1
	for i=1,3 do
		content_batch_size[i+1] = (#content_img)[i]
	end
	content_img = torch.reshape(content_img,content_batch_size)

	local style_img = loadimg(params.style_img,256):float()
	local style_batch_size = torch.LongStorage(4)
	style_batch_size[1] = 1
	for i=1,3 do
		style_batch_size[i+1] = (#style_img)[i]
	end
	style_img = torch.reshape(style_img,style_batch_size)
	
	-- Set up the networks
	local transfer_model = transferNet():float()
	local perception_model, losslayers = prepare_VGG16(params.style_weight,params.content_weight,params.tv_weight,params.protofile,params.modelfile,params.content_layers_t,params.style_layers_t)
	perception_model:float()

	-- Our model will transfer tv and content loss from content image, so we replace those
	-- y represents the values we're trying to optimize towards
	local yc = perception_model:forward(content_img)
	local yg = {}

	for i = 1, #losslayers do
		local lossname = losslayers[i]
		if lossname == "tv" or lossname == "content" then
			yg[i] = yc[i]:clone()
		end
	end

	local y = perception_model:forward(style_img)
	for i = 1, #losslayers do
		local lossname = losslayers[i]
		if lossname == "style" then
			yg[i] = y[i]:clone()
		end
	end
	y = nil
	yc = nil

	-- Our criterion is MSE, we set it and the gradients up here
	local criterion = nn.ParallelCriterion()
	for i = 1, #losslayers do
		criterion:add(nn.MSECriterion())
	end
	criterion:float()
	local optParams, gradParams = transfer_model:getParameters()

	-- Define the loss&gradient function, optimizer's state
	local function feval(optParams)
		gradParams:zero()

		-- Just run the two networks back and forth
		-- Params does not include the perception model's parameters
		-- as we don't need to train the perecption model
		local out_img = transfer_model:forward(content_img)
		local yhat = perception_model:forward(out_img)
		local loss = criterion:forward(yhat,yg)
		local loss_grads = criterion:backward(yhat,yg)
		local perception_grads = perception_model:backward(out_img,loss_grads)
		transfer_model:backward(content_img,perception_grads)

		collectgarbage()
		return loss, gradParams
	end

	local optim_state = {learningRate = 1e-3}

	-- Run the optimization for n iterations
	print('Running optimization with ADAM')
	for t = 1, params.iterations do
		local x, losses = optim.adam(feval, optParams, optim_state)
		print('Iteration number: '.. t ..'; Current loss: '.. losses[1])
	end

	-- Save the model after training
	-- clear the model before saving
	transfer_model:clearState()
	torch.save(params.trained_model, transfer_model)

end

local params = cmd:parse(arg)
main(params)