import contextlib
import os
import time

import torch
from torch.profiler import profile, ProfilerActivity


def evaluate(model, test_loader, device):
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    dt = Profile(cuda=device=='cuda')

    # Warm up
    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float).to(device)
    for i in range(10):
        _ = model(dummy_input)

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            
            with dt:
                output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    top1_err = (1 - correct_1 / len(test_loader.dataset)).item()
    top5_err = (1 - correct_5 / len(test_loader.dataset)).item()

    return top1_err, top5_err, dt.t / len(test_loader.dataset) * 1E3


def evaluate_onnx(ort_session, test_loader):
    correct_1 = 0.0
    correct_5 = 0.0
    dt = Profile(cuda=False)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Warm up
    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    for i in range(10):
        _ = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}

            with dt:
                output = ort_session.run(None, ort_inputs)
            output = torch.from_numpy(output[0])
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    top1_err = (1 - correct_1 / len(test_loader.dataset)).item()
    top5_err = (1 - correct_5 / len(test_loader.dataset)).item()

    return top1_err, top5_err, dt.t / len(test_loader.dataset) * 1E3


def create_torch_profile(model, input, device='cpu'):
    model.to(device)
    input.to(device)
    activities = [ProfilerActivity.CUDA] if type == 'cuda' else [ProfilerActivity.CPU]
    with profile(activities=activities,
            profile_memory=True, record_shapes=True) as prof:
        model(input)

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


def get_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    os.remove('temp.p')
    return size/1e3


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'.
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py
    """

    def __init__(self, t=0.0, cuda=False):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t
        self.cuda = cuda

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()