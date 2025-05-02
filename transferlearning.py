import pandas as pd
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_new = pd.read_csv(NEW_WORKLOAD_CSV)
X_new = df_new[FEATURE_COLUMNS].values
y_time_new = df_new[TARGET_TIME_COL].values.reshape(-1, 1)
y_power_new = (df_new[TARGET_POWER_COL] / 1000.0).values.reshape(-1, 1)

# Scale using profiling scalers
X_new_scaled = feature_scaler.transform(X_new)
y_time_new_scaled = time_scaler.transform(y_time_new)
y_power_new_scaled = power_scaler.transform(y_power_new)

# Split new workload
X_tr_new, X_te_new, y_t_tr_new, y_t_te_new = train_test_split(
    X_new_scaled, y_time_new_scaled, test_size=0.2, random_state=0)
_, _, y_p_tr_new, y_p_te_new = train_test_split(
    X_new_scaled, y_power_new_scaled, test_size=0.2, random_state=0)

# Convert to tensors
X_tr_new_t = to_tensor(X_tr_new)
X_te_new_t = to_tensor(X_te_new)
y_t_tr_new_t = to_tensor(y_t_tr_new)
y_t_te_new_t = to_tensor(y_t_te_new)
y_p_tr_new_t = to_tensor(y_p_tr_new)
y_p_te_new_t = to_tensor(y_p_te_new)

# Transfer learning function
def transfer_learn(base_model_path, scaler_target, X_tr, y_tr, X_te, y_te,
                   epochs, title):
    # Load base model and replace last layer
    model = SimpleNN()
    model.load_state_dict(torch.load(base_model_path))
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last layer parameters
    for param in model.net[-1].parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.MSELoss()
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_te), y_te).item()
            test_losses.append(test_loss)

    # Plot
    plt.figure()
    plt.plot(train_losses, label="TL Train Loss")
    plt.plot(test_losses, label="TL Test Loss")
    plt.title(f"{title} Transfer Learning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Apply transfer learning
model_time_tl = transfer_learn("model_time_reference.pt", time_scaler,
                               X_tr_new_t, y_t_tr_new_t, X_te_new_t, y_t_te_new_t,
                               epochs=20, title="Time")

model_power_tl = transfer_learn("model_power_reference.pt", power_scaler,
                                X_tr_new_t, y_p_tr_new_t, X_te_new_t, y_p_te_new_t,
                                epochs=20, title="Power")

# Save transfer-learned models
torch.save(model_time_tl.state_dict(), "model_time_transfer.pt")
torch.save(model_power_tl.state_dict(), "model_power_transfer.pt")
