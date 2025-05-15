import torch
import torch.nn as nn
import torch.optim as optim

# Define the custom network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.h1 = nn.Linear(3, 5)
        self.h2 = nn.Linear(5, 2)
        self.h3 = nn.Linear(2, 3)
        self.h4 = nn.Linear(3, 2)
        self.h5 = nn.Linear(2, 2)
        self.h6 = nn.Linear(2, 2)
        self.h7 = nn.Linear(2, 1)

        # Assign weights
        self.h1.weight.data = torch.tensor([[0, 0, 1],[0, 1, 0],[1, 0, 0],[1, 1, 0],[0, 1, 1]], dtype=torch.float32)
        self.h2.weight.data = torch.tensor([[1, 1, -1, 0, 0],[0, 0, 1, 1, -1]], dtype=torch.float32)
        self.h3.weight.data = torch.tensor([[1, 1],[1, -1],[1, 2]], dtype=torch.float32)
        self.h4.weight.data = torch.tensor([[1, -1, 0],[0, -1, 1]], dtype=torch.float32)
        self.h5.weight.data = torch.tensor([[0, 1],[1, 0]], dtype=torch.float32)
        self.h6.weight.data = torch.tensor([[1, -1],[1, 1]], dtype=torch.float32)
        self.h7.weight.data = torch.tensor([[1, -1]], dtype=torch.float32)

        # Assign same bias B to layers that require it
        B = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        self.h1.bias.data = B
        self.h2.bias.data = B[:2]
        self.h3.bias.data = B[:3]
        self.h4.bias.data = B[:2]
        self.h5.bias.data = B[:2]
        self.h6.bias.data = B[:2]
        self.h7.bias.data = torch.tensor([0.], dtype=torch.float32)

    def forward(self, input):
        out = torch.relu(self.h1(input))
        out = torch.relu(self.h2(out))
        out = torch.relu(self.h3(out))
        out = torch.relu(self.h4(out))
        out = torch.relu(self.h5(out))
        out = torch.relu(self.h6(out))
        out = self.h7(out)
        return out

# Sample input
x = torch.tensor([[3, 4, 5], [5, 4, 3]], dtype=torch.float32)

# Instantiate and run the model
model = MyNetwork()
output = model(x)
print("Output:\n", output)
