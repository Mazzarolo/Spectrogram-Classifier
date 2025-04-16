import os
import pandas as pd

base_path = "epoch_logs"

epoch_step_data = {}

for folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)

    if os.path.isdir(folder_path) and folder.startswith("version_"):
        csv_path = os.path.join(folder_path, "metrics.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if {"epoch", "step"}.issubset(df.columns) and not df.empty:
                last_epoch = df["epoch"].iloc[-1]
                last_step = df["step"].iloc[-1]
                epoch_step_data[folder] = (last_epoch, last_step)

windows_times_path = "windows_times.csv"

if os.path.exists(windows_times_path):
    windows_df = pd.read_csv(windows_times_path)

    windows_df["last_epoch"] = None
    windows_df["last_step"] = None

    for i, version in enumerate(epoch_step_data.keys()):
        if i < len(windows_df):
            windows_df.loc[i, "last_epoch"] = epoch_step_data[version][0]
            windows_df.loc[i, "last_step"] = epoch_step_data[version][1]

    output_path = "windows_times_and_epochs.csv"

    windows_df.to_csv(output_path, index=False)
    print(f"Arquivo salvo com sucesso em: {output_path}")
else:
    print(f"Arquivo {windows_times_path} não encontrado!")
