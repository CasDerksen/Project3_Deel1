import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# Gebruik de GPU als die er is, anders de CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ==========================================
# 1. Data inladen en opschonen
# ==========================================
df = pd.read_parquet('./prepared_data/forecast_targets_with_weather.parquet')
huis_generation = df[(df['location'] == 'huis') & (df['target_type'] == 'generation')].copy()
huis_generation = huis_generation.dropna(subset=['pow_imputed', 'Windrichting', 'WindSnelheid', 'LuchtVochtigheid'])

# ==========================================
# 2. Voorspel instellingen & Features
# ==========================================
stap_per_dag = 24 * 12               # 288 stappen per dag (elke 5 minuten)
aantal_dagen_input = 7               # aantal dagen voor inputhistorie
input_len = aantal_dagen_input * stap_per_dag

aantal_dagen_forecast = 3            # aantal dagen voorspelling (horizon)
horizon = aantal_dagen_forecast * stap_per_dag

TIME_COL = 'timestamp'   
tijd = pd.to_datetime(huis_generation[TIME_COL])
uur = tijd.dt.hour                                   
dag_van_week = tijd.dt.dayofweek + uur / 24          

huis_generation['uur_sin']  = np.sin(2 * np.pi * uur / 24)
huis_generation['uur_cos']  = np.cos(2 * np.pi * uur / 24)
huis_generation['week_sin'] = np.sin(2 * np.pi * dag_van_week / 7)
huis_generation['week_cos'] = np.cos(2 * np.pi * dag_van_week / 7)
rad = np.deg2rad(huis_generation['Windrichting'])
huis_generation['wind_sin'] = np.sin(rad)
huis_generation['wind_cos'] = np.cos(rad)

FEATURE_COLS = ['pow_imputed', 'uur_sin', 'uur_cos', 'week_sin', 'week_cos', 'Temperatuur', 'GlobaleStraling', 'DuurNeerslag']
TARGET_COL = 'pow_imputed'

TRAIN_FRAC = 0.60                    
VAL_FRAC = 0.25                      

n_features = len(FEATURE_COLS)
target_idx = FEATURE_COLS.index(TARGET_COL)

# ==========================================
# 3. Chronologisch splitsen & Normaliseren
# ==========================================
values = huis_generation[FEATURE_COLS].values

n = len(values)
i_train = int(n * TRAIN_FRAC)
i_val = int(n * (TRAIN_FRAC + VAL_FRAC))

train_raw = values[:i_train]
val_raw = values[i_train:i_val]
test_raw = values[i_val:]

# Normalisatie op basis van de train-set
train_mean = train_raw.mean(axis=0)
train_std = train_raw.std(axis=0)

train = (train_raw - train_mean) / train_std
val = (val_raw - train_mean) / train_std
test = (test_raw - train_mean) / train_std

# ==========================================
# 4. PyTorch Dataset & DataLoader
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_len, horizon, target_idx):
        self.data = data
        self.input_len = input_len
        self.horizon = horizon
        self.target_idx = target_idx
        self.length = len(data) - input_len - horizon + 1
        
    def __len__(self):
        return max(0, self.length)
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.horizon, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

batch_size = 64
train_dataset = TimeSeriesDataset(train, input_len, horizon, target_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Aantal trainingsvoorbeelden: {len(train_dataset)}")

# ==========================================
# 5. Model Definitie (LSTM Forecaster)
# ==========================================
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_dim, horizon, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, horizon)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1] 
        out = self.fc(last_hidden) 
        return out

HIDDEN_DIM = 64
model = LSTMForecaster(n_features=n_features, hidden_dim=HIDDEN_DIM, horizon=horizon).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 6. Training Loop (met opslaan van losses)
# ==========================================
EPOCHS = 60 
epoch_losses = []

print(f"Start training voor {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        if outputs.shape != batch_y.shape:
            outputs = outputs.view_as(batch_y)
            
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Tijd: {elapsed:.2f}s")

# Plot de trainingsloss over de epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', label='Training Loss (MSE)')
plt.title('Trainingsloss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================
# 7. Rolling Average Forecast Validation (in Watts)
# ==========================================
def create_rolling_validation_windows(data, input_len, horizon, step_size):
    windows = []
    for start_idx in range(0, len(data) - input_len - horizon + 1, step_size):
        end_input_idx = start_idx + input_len
        end_horizon_idx = end_input_idx + horizon
        windows.append((start_idx, end_input_idx, end_horizon_idx))
    return windows

val_windows = create_rolling_validation_windows(val, input_len, horizon, step_size=288)
print(f"\nAantal rolling validation stappen: {len(val_windows)}")

target_mean = train_mean[target_idx]
target_std = train_std[target_idx]

model.eval()
maes_in_watts = []
rmses_in_watts = []

with torch.no_grad():
    for start_idx, end_input_idx, end_horizon_idx in val_windows:
        x_win = torch.tensor(val[start_idx:end_input_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        y_true_norm = val[end_input_idx:end_horizon_idx, target_idx]
        
        y_pred_norm = model(x_win)
        if y_pred_norm.shape != torch.tensor(y_true_norm).unsqueeze(0).shape:
            y_pred_norm = y_pred_norm.view_as(torch.tensor(y_true_norm).unsqueeze(0))
            
        y_pred_norm = y_pred_norm.cpu().numpy().squeeze()
        
        y_true_watts = y_true_norm * target_std + target_mean
        y_pred_watts = y_pred_norm * target_std + target_mean
        
        mae = np.mean(np.abs(y_pred_watts - y_true_watts))
        rmse = np.sqrt(np.mean((y_pred_watts - y_true_watts) ** 2))
        
        maes_in_watts.append(mae)
        rmses_in_watts.append(rmse)

print(f"Gemiddelde MAE over alle rolling windows: {np.mean(maes_in_watts):.2f} Watt")
print(f"Gemiddelde RMSE over alle rolling windows: {np.mean(rmses_in_watts):.2f} Watt")