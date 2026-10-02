from .celeba import build_celeba_loader
from .helen import build_helen_loader

__all__ = ["build_celeba_loader", "build_helen_loader", "build_dataloader"]


def build_dataloader(cfg: dict, split: str):
    dataset = cfg.get("data", {}).get("dataset", "celeba").lower()
    if dataset == "helen":
        return build_helen_loader(cfg, split)
    return build_celeba_loader(cfg, split)
