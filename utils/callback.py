# utils/callback.py
import datetime
import os
import torch
import matplotlib
# 确保在无图形界面的环境（如服务器）下正常工作
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision # 需要 torchvision 来创建图像网格

class LossHistory():
    """ 记录训练和验证过程中的损失和指标，并保存到日志文件和TensorBoard。 """
    def __init__(self, log_dir, model=None, input_shape=None):
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        # --- 修改: 初始化所有指标列表 ---
        self.train_acc = []
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.train_reid_loss = []
        self.train_rec_loss = []
        self.train_lsc_loss = []
        self.train_g_disc_loss = []
        self.val_reid_loss = []

        os.makedirs(self.log_dir, exist_ok=True)
        try:
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard log directory: {self.log_dir}")
        except Exception as e:
            print(f"Error initializing SummaryWriter: {e}")
            self.writer = None

        if model is not None and input_shape is not None and self.writer:
            # ... (add_graph logic, 保持注释或改进) ...
            print("LossHistory: Skipping add_graph for complex model.")

    def append_loss(self, epoch, metrics):
        """
        记录一个epoch的各项指标。

        Args:
            epoch (int): 当前的epoch数 (从0开始)。
            metrics (dict): 包含所有需要记录指标的字典。
                            期望键值: 'train_acc', 'train_loss', 'val_loss', 'val_acc',
                                     'train_reid_l', 'train_rec_l', 'train_lsc_l', 'train_gdisc_l',
                                     'val_reid_l'
        """
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # --- 修改: 从 metrics 字典中安全地获取指标 ---
        train_acc = metrics.get('train_acc', 0.0)
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        val_acc = metrics.get('val_acc', 0.0)
        train_reid_l = metrics.get('train_reid_l', 0.0)
        train_rec_l = metrics.get('train_rec_l', 0.0)
        train_lsc_l = metrics.get('train_lsc_l', 0.0)
        train_gdisc_l = metrics.get('train_gdisc_l', 0.0)
        val_reid_l = metrics.get('val_reid_l', val_loss) # 如果没有提供，用总验证损失代替

        # --- 修改: 更新正确的内部列表 ---
        self.train_acc.append(float(train_acc))
        self.train_loss.append(float(train_loss))
        self.val_loss.append(float(val_loss))
        self.val_acc.append(float(val_acc))
        self.train_reid_loss.append(float(train_reid_l))
        self.train_rec_loss.append(float(train_rec_l))
        self.train_lsc_loss.append(float(train_lsc_l))
        self.train_g_disc_loss.append(float(train_gdisc_l))
        self.val_reid_loss.append(float(val_reid_l))

        # --- 修改: 写入所有指标到日志文件 ---
        log_items = {
            "epoch_train_acc.txt": self.train_acc[-1],
            "epoch_train_loss.txt": self.train_loss[-1],
            "epoch_val_loss.txt": self.val_loss[-1],
            "epoch_val_acc.txt": self.val_acc[-1],
            "epoch_train_reid_loss.txt": self.train_reid_loss[-1],
            "epoch_train_rec_loss.txt": self.train_rec_loss[-1],
            "epoch_train_lsc_loss.txt": self.train_lsc_loss[-1],
            "epoch_train_g_disc_loss.txt": self.train_g_disc_loss[-1],
            "epoch_val_reid_loss.txt": self.val_reid_loss[-1],
        }
        for filename, value in log_items.items():
            try:
                with open(os.path.join(self.log_dir, filename), 'a') as f:
                    f.write(f"{value:.6f}\n")
            except Exception as e:
                print(f"Error writing log file {filename}: {e}")

        # --- 修改: 写入所有指标到TensorBoard ---
        if self.writer:
            try:
                self.writer.add_scalar('Accuracy/train', self.train_acc[-1], epoch)
                self.writer.add_scalar('Accuracy/validation', self.val_acc[-1], epoch)

                self.writer.add_scalar('Loss_Total/train', self.train_loss[-1], epoch)
                self.writer.add_scalar('Loss_Total/validation', self.val_loss[-1], epoch) # 这是基于ReID的val loss

                self.writer.add_scalar('Loss_Train_Components/1_ReID', self.train_reid_loss[-1], epoch)
                self.writer.add_scalar('Loss_Train_Components/2_Reconstruction', self.train_rec_loss[-1], epoch)
                self.writer.add_scalar('Loss_Train_Components/3_Latent_Consistency', self.train_lsc_loss[-1], epoch)
                self.writer.add_scalar('Loss_Train_Components/4_Guidance_Discriminative', self.train_g_disc_loss[-1], epoch)

                # 验证集只记录ReID损失（它等于总验证损失）
                self.writer.add_scalar('Loss_Validation/ReID', self.val_reid_loss[-1], epoch)
            except Exception as e:
                print(f"Error writing to TensorBoard: {e}")

        # --- 绘制并保存损失曲线图 ---
        self.plot_metrics() # 调用绘图函数

    # --- add_images 和 plot_metrics 函数保持不变 (上一版本已更新) ---
    def add_images(self, tag, images, global_step):
         # ... (保持上一版本实现) ...
        if self.writer and images is not None and images.nelement() > 0: # 检查非空
            try:
                # 确保图像范围在 [0, 1] 以便可视化
                if images.min() < -0.1: # 假设范围是 [-1, 1]
                    images = (images + 1.0) / 2.0
                images = torch.clamp(images, 0, 1) # 确保在 [0, 1] 范围内

                # 创建图像网格
                img_grid = torchvision.utils.make_grid(images, nrow=min(4, images.size(0))) # 每行最多显示4张图, 至少1张
                # 添加到 TensorBoard
                self.writer.add_image(tag, img_grid, global_step)
            except Exception as e:
                print(f"Error adding images to TensorBoard (tag: {tag}): {e}")

    def plot_metrics(self):
        # ... (保持上一版本实现，绘制总损失、准确率和各部分训练损失) ...
        iters = range(len(self.train_loss)) # 使用 train_loss 的长度作为迭代次数
        if not iters: return

        plt.figure(figsize=(18, 10))

        # 子图1: 总训练/验证损失
        plt.subplot(2, 3, 1)
        if self.train_loss: plt.plot(iters, self.train_loss, 'r-', linewidth=1.5, label='Train Total Loss')
        if self.val_loss: plt.plot(iters, self.val_loss, 'b-', linewidth=1.5, label='Val Total Loss (ReID)')
        try: # 平滑
            num = min(len(iters) // 2 * 2 -1, 15) if len(iters) > 5 else 5
            if num >= 3:
                 if len(self.train_loss) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.train_loss, num, 3), 'g--', linewidth=1)
                 if len(self.val_loss) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 'c--', linewidth=1)
        except Exception as e: print(f"Loss smoothing failed: {e}")
        plt.title('Total Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(fontsize='small')

        # 子图2: 训练/验证准确率
        plt.subplot(2, 3, 2)
        if self.train_acc: plt.plot(iters, self.train_acc, 'r-', linewidth=1.5, label='Train Accuracy')
        if self.val_acc: plt.plot(iters, self.val_acc, 'b-', linewidth=1.5, label='Val Accuracy')
        try: # 平滑
            num = min(len(iters) // 2 * 2 -1, 15) if len(iters) > 5 else 5
            if num >= 3:
                 if len(self.train_acc) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'g--', linewidth=1)
                 if len(self.val_acc) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), 'c--', linewidth=1)
        except Exception as e: print(f"Acc smoothing failed: {e}")
        plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend(fontsize='small'); plt.ylim(bottom=0.0)

        # 子图3: 训练 Re-ID 损失
        plt.subplot(2, 3, 3)
        if self.train_reid_loss: plt.plot(iters, self.train_reid_loss, 'r-', linewidth=1.5, label='Train Re-ID Loss')
        try: # 平滑
            num = min(len(iters) // 2 * 2 -1, 15) if len(iters) > 5 else 5
            if num >= 3 and len(self.train_reid_loss) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.train_reid_loss, num, 3), 'g--', linewidth=1)
        except Exception as e: print(f"ReID loss smoothing failed: {e}")
        plt.title('Re-ID Loss (Train)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(fontsize='small')

        # 子图4: 训练 重建损失
        plt.subplot(2, 3, 4)
        if self.train_rec_loss: plt.plot(iters, self.train_rec_loss, 'g-', linewidth=1.5, label='Train Rec Loss')
        try: # 平滑
            num = min(len(iters) // 2 * 2 -1, 15) if len(iters) > 5 else 5
            if num >= 3 and len(self.train_rec_loss) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.train_rec_loss, num, 3), 'c--', linewidth=1)
        except Exception as e: print(f"Rec loss smoothing failed: {e}")
        plt.title('Reconstruction Loss (Train)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(fontsize='small')

        # 子图5: 训练 潜在一致性损失
        plt.subplot(2, 3, 5)
        if self.train_lsc_loss: plt.plot(iters, self.train_lsc_loss, 'm-', linewidth=1.5, label='Train LSC Loss')
        try: # 平滑
            num = min(len(iters) // 2 * 2 -1, 15) if len(iters) > 5 else 5
            if num >= 3 and len(self.train_lsc_loss) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.train_lsc_loss, num, 3), 'y--', linewidth=1)
        except Exception as e: print(f"LSC loss smoothing failed: {e}")
        plt.title('Latent Consistency Loss (Train)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(fontsize='small')

        # 子图6: 训练 结构判别损失
        plt.subplot(2, 3, 6)
        if self.train_g_disc_loss: plt.plot(iters, self.train_g_disc_loss, 'c-', linewidth=1.5, label='Train G-Disc Loss')
        try: # 平滑
            num = min(len(iters) // 2 * 2 -1, 15) if len(iters) > 5 else 5
            if num >= 3 and len(self.train_g_disc_loss) >= num: plt.plot(iters, scipy.signal.savgol_filter(self.train_g_disc_loss, num, 3), 'k--', linewidth=1)
        except Exception as e: print(f"G-Disc loss smoothing failed: {e}")
        plt.title('Guidance Disc Loss (Train)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend(fontsize='small')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_metrics_detailed.png"), dpi=150)
        plt.cla(); plt.close("all")

    def close_writer(self):
        """ 关闭TensorBoard写入器 """
        if self.writer:
            try:
                self.writer.close(); print("TensorBoard writer closed.")
            except Exception as e: print(f"Error closing TB writer: {e}")