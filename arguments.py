class TrainArguments:
    net: str
    gpu: bool = False
    batch_size: int = 256
    lr: float = 0.1
    warm: int = 1
    resume: bool = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

            
class TestArguments:
    net: str
    gpu: bool = False
    batch_size: int = 32
    weights: str

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)