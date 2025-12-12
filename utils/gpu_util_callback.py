import time
import threading
import torch
import pytorch_lightning as pl

class PeakGPUMemoryCallback(pl.Callback):
    def __init__(self):
        self.train_epoch_peaks = []

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        self.train_epoch_peaks.append(peak_mem_mb)

        trainer.logger.log_metrics(
            {"train/peak_gpu_memory_mb": peak_mem_mb},
            step=trainer.current_epoch
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["train_peak_gpu_memory_MiB"] = self.train_epoch_peaks
