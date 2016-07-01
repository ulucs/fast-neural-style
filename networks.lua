require 'nngraph'
require 'loadcaffe'

function ResidualLayer(nFilters,ch,cw,sh,sw,ph,pw)
	-- trying to conserve the network definition pattern
	-- input size:  n*nFilters*width*height
	-- output size: n*nFilters*width*height
	return function (input)
		local m1c = nn.SpatialConvolution(nFilters,nFilters,ch,cw,sh,sw,ph,pw)(input)
		local m1n = nn.SpatialBatchNormalization(nFilters)(m1c)
		local m1r = nn.ReLU()(m1n)
		local m2c = nn.SpatialConvolution(nFilters,nFilters,ch,cw,sh,sw,ph,pw)(m1r)
		local m2n = nn.SpatialBatchNormalization(nFilters)(m2c)
		return nn.CAddTable()({m2n,input})
	end
end

function transferNet()
	-- input size:  n*3*w*h
	-- output size: n*3*w*h
	local x = nn.Identity()()
	local c1 = nn.ReLU()(
				nn.SpatialBatchNormalization(32)(
				nn.SpatialConvolution(3,32,9,9,1,1,4,4)(x)))
	local c2 = nn.ReLU()(
				nn.SpatialBatchNormalization(64)(
				nn.SpatialConvolution(32,64,3,3,2,2,1,1)(c1)))
	local c3 = nn.ReLU()(
				nn.SpatialBatchNormalization(128)(
				nn.SpatialConvolution(64,128,3,3,2,2,1,1)(c2)))
	local r1 = ResidualLayer(128,3,3,1,1,1,1)(c3)
	local r2 = ResidualLayer(128,3,3,1,1,1,1)(r1)
	local r3 = ResidualLayer(128,3,3,1,1,1,1)(r2)
	local d1 = nn.ReLU()(
				nn.SpatialBatchNormalization(64)(
				nn.SpatialFullConvolution(128,64,3,3,2,2,1,1,1,1)(r3)))
	local d2 = nn.ReLU()(
				nn.SpatialBatchNormalization(32)(
				nn.SpatialFullConvolution(64,32,3,3,2,2,1,1,1,1)(d1)))
	local d3 = nn.AddConstant(7.5,true)(
				nn.MulConstant(127.5,true)(
				nn.Tanh()(
				nn.SpatialFullConvolution(32,3,9,9,1,1,4,4,0,0)(d2))))
	return nn.gModule({x},{d3})
end

function prepare_VGG16(style_weight,content_weight,tv_weight,protofile,modelfile,content_layers_t,style_layers_t)

	local content_layers = content_layers_t:split(",")
	local style_layers = style_layers_t:split(",")

	local cnn = loadcaffe.load(protofile, modelfile):float()
	local netsize = 128*256*256

	local next_content_idx, next_style_idx = 1, 1
	local netlayers = {}
	local netoutputs = {}
	local losslayers = {}
	netlayers[1] = nn.Identity()()
	-- Output TV_Loss
	print("Setting up TV layer")
	table.insert(losslayers,"tv")
	netoutputs[#netoutputs+1] = nn.MulConstant(tv_weight/(3*256*256))(netlayers[#netlayers])
	for i = 1, #cnn do
		if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
			local layer = cnn:get(i)
			local name = layer.name
			netlayers[#netlayers+1] = layer(netlayers[#netlayers])
			if name == content_layers[next_content_idx] then
				-- Output content losses
				print("Setting up content layer", i, ":", layer.name)
				table.insert(losslayers,"content")
				netoutputs[#netoutputs+1] = nn.MulConstant(content_weight/(netsize/4))(netlayers[#netlayers])
				next_content_idx = next_content_idx + 1
			end
			if name == style_layers[next_style_idx] then
				-- Output style losses
				print("Setting up style layer  ", i, ":", layer.name)
				table.insert(losslayers,"style")
				netoutputs[#netoutputs+1] = nn.MulConstant(style_weight/(netsize/2^(next_style_idx)))(GramMatrix()(netlayers[#netlayers]))
				next_style_idx = next_style_idx + 1
			end
		end
	end
	return nn.gModule({netlayers[1]},netoutputs), losslayers
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end