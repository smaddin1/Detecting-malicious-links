import torch
from json import loads, dumps
from sys import stdout

CON_TOKS_PATH = "./data/data_processed/contoks.json"
TEST_SET_PATH = "./data/data_processed/test.ldjson"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT_TYPE = torch.cuda.FloatTensor if DEVICE == "cuda" else torch.FloatTensor
LONG_TYPE = torch.cuda.LongTensor if DEVICE == "cuda" else torch.LongTensor

class MLP(torch.nn.Module):
    def __init__(self, *, n_features, n_classes, hidden_dim):
        super().__init__()
        self.predict = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features, hidden_dim, device=DEVICE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, device=DEVICE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, n_classes, device=DEVICE)
        )

    def forward(self, X):
        return self.predict(X)

def load_toks(fname):
    tokens = None

    with open(fname, 'r') as file:
        tokens = loads(file.read())

    print("Loaded tokens.")
    stdout.flush()

    return tokens

def load_data(fname):
    dataset = []

    with open(fname, 'r') as file:
        line = file.readline()
        while line:
            dataset.append(loads(line))
            line = file.readline()
    
    print("Loaded dataset.")
    stdout.flush()
    
    return dataset

if __name__ == "__main__":
    model = torch.load("model_out.pkl")
    model.eval()
    print("Loaded model.")
    stdout.flush()

    to_phase_2 = []
    test_set = load_data(TEST_SET_PATH)
    con_toks = load_toks(CON_TOKS_PATH)

    for i, sample in enumerate(test_set):
        values = list(sample.values())
        X = torch.Tensor([values[:-2]]).type(FLOAT_TYPE).to(DEVICE)
        y = torch.Tensor([values[-1]]).type(LONG_TYPE).to(DEVICE)

        y_hat = model(X)
        class_confidence = torch.nn.functional.softmax(y_hat, dim=1)
        class_confidence = class_confidence.cpu().detach().numpy()

        # determines what moves onto phase 2
        if 0.4975 <= class_confidence[0][0] <= 0.5025:
            sample["content"] = con_toks[sample["content"]]
            to_phase_2.append(sample)

        if i % 1000 == 0:
            print(f"Processed {i} samples")

    with open("phase2_data.json", 'w') as file:
        file.write(dumps(to_phase_2))