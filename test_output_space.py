import paddle
from paddle.vision.models import resnet50
from ppdet.modeling.backbones import *
from ppdet.modeling.necks import *
if __name__ == '__main__':
	# 构造一个随机输入
	input_data = paddle.randn([1, 3, 1024, 1024])

	backbone = ResNet()


	# 执行前向计算,模型输入 x = input['image']
	output_backbone = backbone({'image':input_data})

	neck = fpn.FPN(in_channels=output_backbone[1],out_channel=256)
	output_neck = neck(output_backbone)


	# 输出结果形状, list[Tensor(output)]
	print(output_neck[0].shape)
