require 'image'

function preprocess(img)
	local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
	local perm = torch.LongTensor{3,2,1}
	img = img:index(1, perm):mul(256.0)
	mean_pixel = mean_pixel:view(3,1,1):expandAs(img)
	return img:add(-1, mean_pixel)
end

function deprocess(img)
	local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
	local perm = torch.LongTensor{3,2,1}
	mean_pixel = mean_pixel:view(3,1,1):expandAs(img)
	img = img:add(mean_pixel)
	return img:index(1, perm):div(256.0)
end

function loadimg(path, size)
	local img = image.load(path, 3)
	if size then
		img = image.scale(img, size, 'bilinear')
	end
	return preprocess(img):float()
end

function saveimg(img, path)
	local simg = deprocess(img)
	simg = image.minmax{tensor=simg, min=0, max=1}
	image.save(path, simg)
end