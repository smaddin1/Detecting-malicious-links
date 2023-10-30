import torch
from json import loads
from os import system
from sklearn.metrics import classification_report, confusion_matrix
from sys import stdout

TRAIN: bool = False

LOC_TOKS_PATH = "./data/data_processed/loctoks.json"
TLD_TOKS_PATH = "./data/data_processed/tldtoks.json"
CON_TOKS_PATH = "./data/data_processed/contoks.json"
TEST_SET_PATH = "./data/data_processed/test.ldjson"
TRAIN_SET_PATH = "./data/data_processed/train.ldjson"

def load_toks(fname):
    tokens = None

    with open(fname, 'r') as file:
        tokens = loads(file.read())

    return tokens

def load_data(fname):
    dataset = []

    with open(fname, 'r') as file:
        line = file.readline()
        while line:
            dataset.append(loads(line))
            line = file.readline()
    
    return dataset

LOC_TOKS = load_toks(LOC_TOKS_PATH)
TLD_TOKS = load_toks(TLD_TOKS_PATH)
CON_TOKS = load_toks(CON_TOKS_PATH)
print("Loaded tokens.")
stdout.flush()

test_objs = load_data(TEST_SET_PATH)
train_objs = load_data(TRAIN_SET_PATH)
print("Loaded train and test sets.")
stdout.flush()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLOAT_TYPE = torch.cuda.FloatTensor if DEVICE == "cuda" else torch.FloatTensor
LONG_TYPE = torch.cuda.LongTensor if DEVICE == "cuda" else torch.LongTensor
MAX_VALUES = {'geo_loc': 233, 'url_len': 721, 'netloc_len': 71, 'path_len': 594, 'param_len': 138, 'query_len': 655, 'frag_len': 140, 'tld': 1351, 'who_is': 1, 'https': 1, 'label': 1}

def objs_to_loader(objs, batch_size):
    values_raw = [[value for value in obj.values() if not isinstance(value, str)] for obj in objs]
    values_tensor = torch.tensor(values_raw).type(FLOAT_TYPE).to(DEVICE)
    X_set = values_tensor[:, :-2].type(FLOAT_TYPE)
    y_set = values_tensor[:, -1].type(LONG_TYPE)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_set, y_set),
        batch_size=batch_size,
        shuffle=True
    )

test_loader = objs_to_loader(test_objs, len(test_objs))
train_loader = objs_to_loader(train_objs, len(train_objs))
print("Converted raw objects to dataloaders.")
stdout.flush()

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

def train_loop(model, epochs, loss_fxn, optimizer, train_loader, test_loader):
    train_loss = {}
    train_accuracy = {}
    eval_loss = {}
    eval_accuracy = {}

    for epoch in range(epochs):
        train_loss[epoch] = 0
        eval_loss[epoch] = 0

        model = model.train()
        num_correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.type(FLOAT_TYPE).to(DEVICE)
            y_batch = y_batch.type(LONG_TYPE).to(DEVICE)
        
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = loss_fxn(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            class_confidence = torch.nn.functional.softmax(y_hat, dim=1)
            pred_class = torch.max(class_confidence, 1)[1]
            num_correct_train += pred_class.eq(y_batch).sum().item()
            total_train += y_batch.size(dim=0)

            train_loss[epoch] += loss.item() * X_batch.size(axis=0)
    
        model = model.eval()
        num_correct_eval = 0
        total_eval = 0
        report = None
        conf_mat = None

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.type(FLOAT_TYPE).to(DEVICE)
            y_batch = y_batch.type(LONG_TYPE).to(DEVICE)

            y_hat = model(X_batch)
            loss = loss_fxn(y_hat, y_batch)
            eval_loss[epoch] = loss.item() * X_batch.size(axis=0)
            
            # CLASS CONFIDENCE HERE
            class_confidence = torch.nn.functional.softmax(y_hat, dim=1)
            
            pred_class = torch.max(class_confidence, 1)[1]
            pred_np = pred_class.cpu().detach().numpy()
            batch_np = y_batch.cpu().detach().numpy()
            report = classification_report(batch_np, pred_np)
            conf_mat = confusion_matrix(batch_np, pred_np)
            num_correct_eval += pred_class.eq(y_batch).sum().item()
            total_eval += y_batch.size(dim=0)

        train_loss[epoch] /= total_train
        eval_loss[epoch] /= total_eval

        train_accuracy[epoch] = num_correct_train / total_train
        eval_accuracy[epoch] = num_correct_eval / total_eval

        print(f"Epoch #{epoch+1} stats:")
        # print(f"\tTraining loss: {train_loss[epoch]}")
        # print(f"\tTrain accuracy: {train_accuracy[epoch] * 100:.2f}%")
        # print(f"\tEvaluation loss: {eval_loss[epoch]}")
        # print(f"\tTest accuracy: {eval_accuracy[epoch] * 100:.2f}%")
        #avg_dict(results, num_results)
        print(report)
        print(conf_mat)
        system("mv model_out_curr.pkl model_out_prev.pkl")
        torch.save(model, "model_out_curr.pkl")
        #print(dumps(results, indent=2))
        print()

    pack_info = lambda td, ed: [(t, e) for t, e in zip(td.values(), ed.values())]
    return pack_info(train_loss, eval_loss), pack_info(train_accuracy, eval_accuracy)

if TRAIN:
    EPOCHS = 25

    model = MLP(
        n_features=train_loader.dataset[0][0].size(dim=0),
        n_classes=2,
        hidden_dim=237
    ).to(DEVICE)

    train_args = {
        "model": model,
        "epochs": EPOCHS,
        "loss_fxn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.SGD(
            model.parameters(),
            lr=0.0001,
            momentum=0.99
        ),
        "train_loader": train_loader,
        "test_loader": test_loader
    }

    print("Training the model...")
    stdout.flush()
    try:
        loss, accuracy = train_loop(**train_args)
    finally:
        torch.save(model, "model_out_stop.pkl")
else:
    model = torch.load("model_out.pkl")
    model.eval()

    num_correct_eval = 0
    total_eval = 0
    eval_loss = 0
    loss_fxn = torch.nn.CrossEntropyLoss()
    report = None
    conf_mat = None
    to_phase_2 = []

    """to_phase_2 = []
    test_set = load_data(TEST_SET_PATH)
    con_toks = load_toks(CON_TOKS_PATH)

    for sample in test_set:
        values = list(sample.values())
        X = torch.Tensor(values[:-2]).type(FLOAT_TYPE).to(DEVICE)
        y = torch.Tensor([values[-1]]).type(LONG_TYPE).to(DEVICE)

        y_hat = model(X)
        class_confidence = torch.nn.functional.softmax(y_hat, dim=1)
        class_confidence = class_confidence."""

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.type(FLOAT_TYPE).to(DEVICE)
        y_batch = y_batch.type(LONG_TYPE).to(DEVICE)

        print(X_batch.shape)
        y_hat = model(X_batch)
        loss = loss_fxn(y_hat, y_batch)
        eval_loss = loss.item() * X_batch.size(axis=0)

        # CLASS CONFIDENCE HERE
        class_confidence = torch.nn.functional.softmax(y_hat, dim=1)

        # determines what moves onto phase 2
        for i, (c0, _) in enumerate(class_confidence.cpu().detach().numpy()):
            if 0.4975 <= c0 <= 0.5025:
                reconstructed = X_batch[i].cpu().detach().numpy().tolist()
                reconstructed += []
                reconstructed += y_batch[i].cpu().detach().numpy().tolist()
                to_phase_2.append(reconstructed)

        pred_class = torch.max(class_confidence, 1)[1]
        pred_np = pred_class.cpu().detach().numpy()
        batch_np = y_batch.cpu().detach().numpy()
        report = classification_report(batch_np, pred_np)
        conf_mat = confusion_matrix(batch_np, pred_np)
        num_correct_eval += pred_class.eq(y_batch).sum().item()
        total_eval += y_batch.size(dim=0)

    eval_loss /= total_eval
    eval_accuracy = num_correct_eval / total_eval
    print(eval_accuracy)
    print(report)
    print(conf_mat)
    print(to_phase_2)