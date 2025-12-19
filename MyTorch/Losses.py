class L1Loss:
    def __init__(self):
        pass

    def __call__(self, src, target):
        return abs(src - target).mean()

__all__ = ["L1Loss"]