# 爱用抠图专业训练工具

# 数据集准备
1. 原图与抠好的蒙版图
2. 将原图与抠好的蒙版图按照以下结构保存起来
   原图 -> origins
   蒙版图 -> masks


# 开始训练
1. `cd DIS/IS-Net`
2. 修改第 `DIS/IS-Net/train_valid_inference_main.py` 文件的 第 663 行的数据集代码
   ```python
   # 训练集
   dataset_demo = {"name": "ay-data", # 你数据集的名称
                 "im_dir": "/root/data/train", # 数据集所在目录
                 "gt_dir": "/root/data/train_mask", # mask 所在目录
                 "im_ext": ".jpg", # 原图图片后缀
                 "gt_ext": ".png", # mask 图片后缀
                 "cache_dir":"/root/autodl-tmp/cache/train"} # 缓存路径，请至少准备 1.1 * (你的图片数量) MB 大小的存储空间
    # 测试集，强烈推荐准备一份测试集数据，如果没有可以将 20% 的训练集数据抽离出来
    dataset_val_demo = {"name": "ay-data",
                 "im_dir": "/root/data/val",
                 "gt_dir": "/root/data/val_mask",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir":"/root/autodl-tmp/cache/val"}
   ```

3. 运行 `python train_valid_inference_main.py` 观察运行正常结束
4. 训练中最好的模型将会被保存在 `saved_models` 下
