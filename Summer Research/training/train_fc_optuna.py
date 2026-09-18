import torch
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import os
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IN_FEATURES = ['beta', 'Av', 'F115', 'F150', 'F277', 'F444']
OUT_FEATURES = ['flag']
SEED = 42

# HYPERPARAMETERS
EPOCHS = 1000
RUN = True
torch.manual_seed(SEED)
cwd = Path.cwd()
table_dir = cwd / 'Summer Research' / 'train val test tables'
print(table_dir)

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
table_dfs = []
for table in table_dir.rglob('*'):
    if table.name == 'test.csv':
        continue
    table_dfs.append(pd.read_csv(table, index_col=0))

train_df = pd.concat(table_dfs[0:-2])
val_df = pd.concat(table_dfs[-2:])

df = pd.concat([train_df, val_df], keys=["train", "val"])
df = df[IN_FEATURES + OUT_FEATURES]
df.dropna(inplace=True)

fluxes = IN_FEATURES[2:]
df = df[~df[IN_FEATURES].eq(-99).any(axis=1)]
df[fluxes] = df[fluxes].clip(lower=0)

train_df = df.xs('train').copy()
val_df = df.xs('val').copy()

train_df[fluxes] = np.log1p(train_df[fluxes])
val_df[fluxes] = np.log1p(val_df[fluxes])

scaler = RobustScaler().set_output(transform="pandas")
X_train_df = scaler.fit_transform(train_df[IN_FEATURES])
X_val_df = scaler.transform(val_df[IN_FEATURES])

X_train = torch.tensor(train_df[IN_FEATURES].values)
X_val = torch.tensor(val_df[IN_FEATURES].values)
y_train = torch.tensor(train_df['flag'].values)
y_val = torch.tensor(val_df['flag'].values)

train_data = torch.utils.data.TensorDataset(X_train, y_train)
val_data = torch.utils.data.TensorDataset(X_val, y_val)

# ==========================================
# 2. DEFINE SYSTEM ARCHITECTURE
# ==========================================
def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    in_features = len(IN_FEATURES)
    
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(out_features))
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
        
    layers.append(nn.Linear(in_features, 2 * len(OUT_FEATURES)))
    return nn.Sequential(*layers)

def objective(trial):
    torch.set_num_threads(1) 
        
    model = define_model(trial).to(DEVICE)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])

    kwargs = {"lr": lr}
    if optimizer_name == "AdamW":
        kwargs["weight_decay"] = trial.suggest_float("adamw_weight_decay", 1e-5, 1e-1, log=True)
    elif optimizer_name == "SGD":
        kwargs["momentum"] = trial.suggest_float("sgd_momentum", 0.0, 0.99)

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **kwargs)

    w1 = trial.suggest_float("w1", 1, 3)
    w2 = trial.suggest_float("w2", 1, 3)
    weights = torch.tensor([w1, w2]).to(DEVICE)
    loss_func = nn.CrossEntropyLoss(weight=weights)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    for epoch in range(EPOCHS):
        # Training loop
        model.train()
        running_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if data.ndim == 1:
                data = data.unsqueeze(1)
            data = data.float()
            outputs = model(data)
            loss = loss_func(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

        # Validation loop
        model.eval()
        running_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                if data.ndim == 1:
                    data = data.unsqueeze(1)
                data = data.float()
                outputs = model(data)
                loss = loss_func(outputs, labels.long())
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).float().sum()

            val_loss = running_loss / len(val_loader)
            trial.set_user_attr(f"val_loss_epoch_{epoch}", val_loss)

            accuracy = (correct / len(val_data)).item()
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy

# ==========================================
# 3. PROTECTED MULTIPROCESSING EXECUTION
# ==========================================
if __name__ == '__main__':
    if RUN:
        # Display scaling metrics to terminal before initiating background tasks
        stats_df1 = pd.DataFrame({'Median_Train': X_train_df.median(), 'IQR_Train': X_train_df.quantile(0.75) - X_train_df.quantile(0.25)})
        stats_df2 = pd.DataFrame({'Median_Val': X_val_df.median(), 'IQR_Val': X_val_df.quantile(0.75) - X_val_df.quantile(0.25)})
        print(pd.concat([stats_df1, stats_df2], axis=1).round(4))

        # Adjust the log level to INFO so you can monitor background worker completion status
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # Utilize an SQLite database storage system to avoid process overlap clashes
        study = optuna.create_study(
            direction="maximize", 
            pruner=optuna.pruners.MedianPruner(),
        )

        print("\n--- Starting Parallel Optimization Study ---")
        study.optimize(objective, n_trials=20, n_jobs=10)

        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
