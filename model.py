import torch
import torch.nn as nn


class LinearRegression(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_in, n_out)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, train_dl, test_dl, val_dl=None, epochs: int = 10, lr: float = 1e-4, weight_decay: float = 1e-4):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train': [], 'validation': [], 'test': []}
    for epoch in range(epochs):
        print(f'\n-------- Epoch {epoch + 1} --------')
        # Train model
        model.train()
        acc = 0.
        with torch.enable_grad():
            for batch, (X, y) in enumerate(train_dl):
                # Forward pass
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                acc += (y_hat.argmax(1) == y).type(torch.float).sum().item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f'[{batch}] loss : {loss.item() : >6f}')

        history['train'].append(acc / len(train_dl.dataset))
        print(f'\n\ttrain accuracy {history["train"][epoch]}')

        # Validate and test model
        model.eval()
        with torch.no_grad():
            if val_dl is not None:
                acc = 0.
                for X, y in val_dl:
                    y_hat = model(X)
                    acc += (y_hat.argmax(1) == y).type(torch.float).sum().item()

                history['validation'].append(acc / len(val_dl.dataset))
                print(f'\tvalidation accuracy {history["validation"][epoch]}')

            acc = 0.
            for X, y in test_dl:
                y_hat = model(X)
                acc += (y_hat.argmax(1) == y).type(torch.float).sum().item()

        history['test'].append(acc / len(test_dl.dataset))
        print(f'\ttest accuracy {history["test"][epoch]}')

        torch.save(model, f=f'models/{model.__class__.__name__}/epoch-{epoch}.pt')

    return history


def test_model(model: nn.Module, test_ds) -> float:
    model.eval()

    acc = 0.
    with torch.no_grad():
        for X, y in test_ds:
            y_hat = model(X)
            acc += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    return acc / len(test_ds.dataset)
