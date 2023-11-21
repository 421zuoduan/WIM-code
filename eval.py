import torch

# define the accuracy evaluation method
def evaluate_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        accuracy = float(num_correct / num_samples)
        print('Number of correct sample %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * accuracy))
        return accuracy