import argparse

import torch

from conf import settings
from utils import get_network, get_test_dataloader


def evaluate(model, test_loader, device, verbose=True):
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            if verbose:
                print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            image = image.to(device)
            label = label.to(device)
            if verbose and device == 'cuda':
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


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
    
    if verbose:
        print("Top 1 err:", top1_err)
        print("Top 5 err:", top5_err)
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    return top1_err, top5_err


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    net.load_state_dict(torch.load(args.weights))
    device = "cuda" if args.gpu else "cpu"

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
    )

    evaluate(model=net, test_loader=test_loader, device=device, verbose=True)