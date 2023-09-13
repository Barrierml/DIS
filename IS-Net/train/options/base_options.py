import os.path as osp
import os


class parser(object):
    def __init__(self):
        self.name = "training_cloth_segm_dig"  # Expriment name
        self.image_folder = "/root/autodl-tmp/train/"  # image folder path
        self.df_path = "/root/autodl-tmp/train.csv"  # label csv path
        self.distributed = True  # True for multi gpu training
        self.isTrain = True

        self.fine_width = 192 * 5
        self.fine_height = 192 * 5

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 7  # 12
        self.nThreads = 3  # 3
        self.max_dataset_size = float("inf")

        self.serial_batches = False
        self.continue_train = True
        if self.continue_train:
            self.unet_checkpoint = osp.join(osp.join("results", self.name), 'checkpoints/itr_00001000_dis.pth')

        self.save_freq = 1000
        self.print_freq = 10
        self.image_log_freq = 100

        self.iter = 100000
        self.lr = 0.0001
        self.clip_grad = 5

        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)