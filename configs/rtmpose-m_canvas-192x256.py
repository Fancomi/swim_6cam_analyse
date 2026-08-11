# Plan C 的画布 RTMPose-m 纯推理配置（自包含，不依赖 .exp 实验目录）。
#
# 与 configs/rtmpose-m_swim-256x192.py（Plan A，原相机版）的唯一区别是训练数据：
# 这版在画布伪标签上微调，因此直接吃画布坐标的框，不需要 mesh 映射回原相机。
# 模型结构与输入尺寸完全相同（192x256），所以两者可以互换验证。
#
# 训练配置见 .exp/configs/rtm_canvas_192x256.py（含数据集与增广），
# 这里只保留 init_model / inference_topdown 需要的部分。
# 关键点定义（COCO17 顺序、骨架、翻转对）由权重的 meta.dataset_meta 提供。

default_scope = 'mmpose'

codec = dict(
    type='SimCCLabel', input_size=(192, 256), sigma=(4.9, 5.66),
    simcc_split_ratio=2.0, normalize=False, use_dark=False)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    backbone=dict(
        # CSPNeXt 定义在 mmdet 中，_scope_ 让 mmpose 的注册表跨库查找
        _scope_='mmdet', type='CSPNeXt', arch='P5',
        deepen_factor=0.67, widen_factor=0.75, out_indices=(4, ),
        expand_ratio=0.5, channel_attention=True,
        norm_cfg=dict(type='SyncBN'), act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead', in_channels=768, out_channels=17,
        input_size=codec['input_size'], in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.0,
            drop_path=0.0, act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        loss=dict(type='KLDiscretLoss', use_target_weight=True,
                  beta=10.0, label_softmax=True),
        decoder=codec),
    # flip_test：左右翻转推理后取平均，约 2 倍 GPU 前向换更稳的关键点
    test_cfg=dict(flip_test=True))

# inference_topdown 取 test_dataloader.dataset.pipeline 构建预处理：
# 按 bbox 算中心/尺度 -> 仿射采样到 192x256 -> 打包成模型输入
test_dataloader = dict(dataset=dict(pipeline=[
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]))
