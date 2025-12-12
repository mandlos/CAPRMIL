import pytorch_lightning as pl

class TrainableParamsCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        n_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        pl_module._trainable_params = n_params  # store on module

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["trainable_params"] = pl_module._trainable_params
