from torch import nn

class FutureToPastFF(nn.Module):

    def __init__(self, model, start, step, lambda_val, horizon):

        self.start = start
        self.step = step
        self.lambda_val = lambda_val

        self.model = model
        
        self.ff = []
        self.num_ff = np.arange(start + step, step, 1)
        for idx in self.num_ff:
            self.ff.append(nn.Sequential(nn.Linear(int(horizon * idx), 50), nn.Linear(50, int(horizon * idx))))

        self.ff = nn.ModuleList(self.ff)

    def forward(self, x):

        x, x_ss = x

        y = self.model(x)

        for idx, xs in enumerate(x_ss):
            ys = self.model(xs)
            ys = self.ff[idx](ys)

            x[:, -ys.shape[-1]:] += self.lambda_val * ys
        
        return x

if __name__ == "__main__":

    horizon = 720

    start, step, lambda_val = 0.1, 0.05, 0.1
    model = FutureToPastFF(
