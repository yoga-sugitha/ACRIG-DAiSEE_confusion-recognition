# train_utils/model_loading.py
import os
from omegaconf import DictConfig, OmegaConf
from modules.lightning_module import LightningModule

def load_best_model(trainer, cfg, class_names, fallback_model):
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("⚠ No checkpoint found. Using current model.")
        return fallback_model

    # Optional: Validate num_classes match
    expected_classes = len(class_names)
    # You'd need to store num_classes in checkpoint hparams to validate
    # For now, just try to load and catch error

    try:
        model = LightningModule.load_from_checkpoint(
            ckpt_path,
            model_name=cfg.model.name,
            model_hparams=OmegaConf.to_container(cfg.model.hparams, resolve=True),
            optimizer_name=cfg.optimizer.name,
            optimizer_hparams=OmegaConf.to_container(cfg.optimizer.hparams, resolve=True),
            class_names=class_names,
            map_location='cpu'
        )
        return model
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"⚠ Checkpoint class mismatch: {e}")
            print("⚠ Falling back to current model weights.")
            return fallback_model
        else:
            raise