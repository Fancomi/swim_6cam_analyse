# RTMPose-m 游泳关键点（COCO17）纯推理配置。
#
# 从训练 work_dir 里导出的完整配置中裁剪而来，只保留 init_model /
# inference_topdown 实际需要的三部分：模型结构、SimCC 解码器、测试期数据
# 流水线。训练相关内容（数据集路径、优化器、调度器、EMA、评测器、
# backbone 的 init_cfg 预训练下载链接）全部去掉——推理时权重来自
# --pose-checkpoint，配置里的 init_cfg 会触发无意义的联网下载。
#
# 关键点定义（COCO17 顺序、骨架、翻转对）由权重文件的 meta.dataset_meta
# 提供，init_model 会优先读取它，因此这里不需要 dataset_type/metainfo。

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
    # flip_test：左右翻转推理后取平均，约 2 倍耗时换取更稳的关键点，
    # 翻转对索引取自权重的 dataset_meta.flip_indices
    test_cfg=dict(flip_test=True))

# inference_topdown 直接取 test_dataloader.dataset.pipeline 构建预处理：
# 按 bbox 算中心/尺度 -> 仿射采样到 192x256 -> 打包成模型输入
test_dataloader = dict(dataset=dict(pipeline=[
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]))
