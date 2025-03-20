import torch
import torch.nn as nn

class FakeClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(FakeClassifier, self).__init__()
        self.conv = nn.Conv2d(1, num_classes, kernel_size=3, padding=1)
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_class_guidance(self, x, y):
        """
        Computes gradient guidance for classifier-based diffusion.
        - x: The current noisy sample.
        - y: Target class index.
        """
        x = x.clone().requires_grad_(True)
        logits = self.forward(x)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = log_probs[:, y].sum()

        grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        return grad