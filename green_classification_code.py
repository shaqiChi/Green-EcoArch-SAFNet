# =============================================
# Green vs Non-Green Architecture Classification
# Real Implementation (PyTorch + LIME + SHAP)
# =============================================

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import torch, shap, lime.lime_tabular
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# ----- Load and Preprocess -----
data = pd.read_csv("green_architecture_dataset.csv")
label_encoders = {}
for col in data.select_dtypes(include='object'):
    le = LabelEncoder(); data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
data.fillna(data.mean(numeric_only=True), inplace=True)

# ----- Feature Groups -----
energy = ["energy_rating", "carbon_emission", "solar_panel"]
ecotech = ["smart_meter", "battery_system", "green_cert"]
design = ["roof_type", "window_glazing"]
context = ["location_score", "urban_density"]
all_features = energy + ecotech + design + context

X = data[all_features]
y = data["label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- Split -----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# ----- Info Gain -----
info_gain = mutual_info_classif(X_train, y_train)
info_df = pd.DataFrame({"Feature": all_features, "InfoGain": info_gain})

# ----- Model -----
class Net(nn.Module):
    def __init__(self, dim): super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.seq(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(X_train.shape[1]).to(device)
Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1).to(device)
Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
yte = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1).to(device)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

train_losses, val_accs = [], []
for epoch in range(30):
    model.train(); loss_sum = 0
    for xb, yb in train_loader:
        out = model(xb)
        loss = loss_fn(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
    model.eval()
    with torch.no_grad():
        preds = model(Xte).round()
        acc = (preds.eq(yte)).sum().item() / len(yte)
    train_losses.append(loss_sum / len(train_loader)); val_accs.append(acc)

# ----- Eval -----
model.eval()
with torch.no_grad():
    y_pred = model(Xte).round().cpu().numpy().astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

# ----- Plots -----
# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Green', 'Green'],
            yticklabels=['Non-Green', 'Green'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show()

# 2. Val Accuracy
plt.plot(val_accs, label="Val Accuracy", color='green')
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Epoch-wise Validation Accuracy")
plt.tight_layout(); plt.show()

# 3. Info Gain
sns.barplot(data=info_df.sort_values("InfoGain", ascending=False), x="InfoGain", y="Feature", palette="crest")
plt.title("Information Gain by Feature")
plt.tight_layout(); plt.show()

# 4. SHAP
explainer = shap.Explainer(model, torch.tensor(X_train, dtype=torch.float32))
shap_values = explainer(torch.tensor(X_test, dtype=torch.float32))
shap.summary_plot(shap_values.values, features=X_test, feature_names=all_features)

# 5. LIME
lime_exp = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=all_features, class_names=['Non-Green', 'Green'], mode='classification')
lime_result = lime_exp.explain_instance(X_test[0], lambda x: model(torch.tensor(x, dtype=torch.float32).to(device)).detach().cpu().numpy())
lime_result.show_in_notebook(show_table=True)

# 6. Attention Weights (placeholder demo)
attention_weights = np.random.rand(4, 30)  # Replace with model output
plt.imshow(attention_weights, aspect="auto", cmap="viridis")
plt.colorbar(); plt.yticks(np.arange(4), ["Energy", "Eco-Tech", "Design", "Context"])
plt.xlabel("Steps"); plt.ylabel("Feature Group"); plt.title("Attention Weights")
plt.tight_layout(); plt.show()
