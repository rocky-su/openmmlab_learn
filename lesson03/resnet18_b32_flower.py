# coding:utf-8
"""
#author: learner
"""

"""
构建配置⽂件可以使⽤继承机制，从 configs/__base__中继承ImageNet预训练的任何模型

为了适配数据集flower这个 5分类数据集，需要修改配置⽂件中模型对应的 head 和 num_classes
"""
_base_ = ['../_base_/models/resnet18.py',
		  '../_base_/datasets/imagenet_bs32.py',
		  '../_base_/default_runtime.py'
];

model = dict(
	head=dict(
		num_classes=5,
		topk = (1,)
));


data = dict(
	# 根据实验环境调整每个batch_size和workers数量
	samples_per_gpu = 32,
	workers_per_gpu = 2,
	# 指定训练集路径
	train = dict(
		data_prefix = 'data/',
		ann_file = 'data/flower_dataset_train_data/train.txt',
		classes = 'data/flower_dataset_train_data/classes.txt'
	),
	val = dict(
		data_prefix = 'data/',
		ann_file = 'data/flower_dataset_train_data/val.txt',
		classes = 'data/flower_dataset_train_data/classes.txt'
	)
)

"""
学习率
模型微调的策略与从头开始训练的策略差别很⼤。微调⼀般会要求更⼩的学习率和更少的训练周期。
依旧是在resnet18_b16_flower.py⽂件中进⾏修改。
"""
# 优化器
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip = None);

# 学习率策略
lr_config = dict(
	policy='step',
	step=[1]
);

# 定义评估方法
evaluation = dict(metric_options={"topk": (1, )})

# 循环100次
runner = dict(type='EpochBasedRunner', max_epochs=100)

# 预训练模型
load_from = '/home/ml_ocr_openmmlab/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'

