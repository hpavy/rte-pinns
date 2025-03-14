# Les fonctions utiles ici
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from model import GNN
import torch
import numpy as np
from torch.utils.data import Dataset


def write_csv(data, path, file_name):
    dossier = Path(path)
    df = pd.DataFrame(data)
    # Créer le dossier si il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    df.to_csv(path + file_name)


def read_csv(path):
    return pd.read_csv(path)


class GraphDataset(Dataset):
    def __init__(self, X_full, U_full, nb_neighbours):
        self.X_full = X_full
        self.U_full = U_full
        time_ = X_full[:, 2].unique()[5]
        self.nb_time = X_full[:, 2].unique().shape[0]
        self.delta_t = (X_full[:, 2].unique()[1:] - X_full[:, 2].unique()[:-1]).mean()
        X = X_full[X_full[:, 2] == time_][:, :2]
        indices = torch.argsort(X[:, 0]*1.1 + X[:, 1]*1.5)
        self.X_sort = X[indices]
        # Broadcasting de génie
        diff = self.X_sort[:, None, :] - self.X_sort[None, :, :]
        # Calcul des distances euclidiennes
        distances = torch.sqrt((diff ** 2).sum(dim=2))
        self.edge_neighbours = torch.empty((self.X_sort.shape[0], nb_neighbours), dtype=torch.long)
        edge_attributes = torch.empty((self.X_sort.shape[0], nb_neighbours, 3))
        for i in range(self.X_sort.shape[0]):
            self.edge_neighbours[i, :] = torch.topk(distances[i, :], nb_neighbours + 1, largest=False)[1][1:]
            for k, neighbour in enumerate(self.edge_neighbours[i, :]):
                neighbour = neighbour.item()
                # print(self.X_sort[i,0]-self.X_sort[neighbour, 0])
                # print(i, neighbour)
                # print(self.X_sort[i,0]-self.X_sort[neighbour, 0])
                # print(self.X_sort[i,1]-self.X_sort[neighbour, 1])
                # print(torch.sqrt(torch.sum((self.X_sort[i, :]-self.X_sort[neighbour, :])**2)))
                edge_attributes[i, k, :] = torch.stack((self.X_sort[i,0]-self.X_sort[neighbour, 0], self.X_sort[i,1]-self.X_sort[neighbour, 1], torch.sqrt(torch.sum((self.X_sort[i]-self.X_sort[neighbour])**2))))
        self.edge_attributes = edge_attributes

    def __len__(self):
        return self.X_full[:, 2].unique().shape[0] - 1

    def __getitem__(self, idx):
        time_ = self.X_full[:, 2].unique()[idx]
        masque_time = self.X_full[:, 2]==time_
        X = self.X_full[masque_time][:, :2]
        indices = torch.argsort(X[:, 0]*1.1 + X[:, 1]*1.5)
        U_sort = self.U_full[masque_time][indices]

        # le next time
        time_n = self.X_full[:, 2].unique()[idx + 1]
        masque_time_n = self.X_full[:, 2]==time_n
        X_n = self.X_full[masque_time_n][:, :2]
        indices_n = torch.argsort(X_n[:, 0]*1.1 + X_n[:, 1]*1.5)
        U_sort_n = self.U_full[masque_time_n][indices_n]
        return U_sort, U_sort_n


def charge_data(hyper_param, param_adim):
    x_full, y_full, t_full = [], [], []
    u_full, v_full, p_full = [], [], []
    x_norm_full, y_norm_full, t_norm_full = (
        [],
        [],
        [],
    )
    u_norm_full, v_norm_full, p_norm_full = [], [], []
    H_numpy = np.array(hyper_param["H"])
    f_numpy = 0.5 * (H_numpy / hyper_param["m"]) ** 0.5
    f = np.min(f_numpy)
    df = pd.read_csv("data/" + hyper_param["file"][0])
    df_modified = df.loc[
        (df["Points:0"] >= hyper_param["x_min"])
        & (df["Points:0"] <= hyper_param["x_max"])
        & (df["Points:1"] >= hyper_param["y_min"])
        & (df["Points:1"] <= hyper_param["y_max"])
        & (df["Time"] > hyper_param["t_min"])
        & (df["Points:2"] == 0.0),
        :,
    ].copy()

    # Adimensionnement
    x_full.append(
        torch.tensor(df_modified["Points:0"].to_numpy(), dtype=torch.float32)
        / param_adim["L"]
    )
    y_full.append(
        torch.tensor(df_modified["Points:1"].to_numpy(), dtype=torch.float32)
        / param_adim["L"]
    )
    time_without_modulo = df_modified["Time"].to_numpy() - hyper_param['t_min']
    # time_with_modulo = hyper_param['t_min'] + time_without_modulo % (1/f)
    t_full.append(
        torch.tensor(time_without_modulo, dtype=torch.float32)
        / (param_adim["L"] / param_adim["V"])
    )
    u_full.append(
        torch.tensor(df_modified["Velocity:0"].to_numpy(), dtype=torch.float32)
        / param_adim["V"]
    )
    v_full.append(
        torch.tensor(df_modified["Velocity:1"].to_numpy(), dtype=torch.float32)
        / param_adim["V"]
    )
    p_full.append(
        torch.tensor(df_modified["Pressure"].to_numpy(), dtype=torch.float32)
        / ((param_adim["V"] ** 2) * param_adim["rho"])
    )
    # les valeurs pour renormaliser ou dénormaliser
    mean_std = {
        "u_mean": torch.cat([u for u in u_full], dim=0).mean(),
        "v_mean": torch.cat([v for v in v_full], dim=0).mean(),
        "p_mean": torch.cat([p for p in p_full], dim=0).mean(),   # On ajoute la pression du bord
        "x_mean": torch.cat([x for x in x_full], dim=0).mean(),
        "y_mean": torch.cat([y for y in y_full], dim=0).mean(),
        "t_mean": torch.cat([t for t in t_full], dim=0).mean(),
        "x_std": torch.cat([x for x in x_full], dim=0).std(),
        "y_std": torch.cat([y for y in y_full], dim=0).std(),
        "t_std": torch.cat([t for t in t_full], dim=0).std(),
        "u_std": torch.cat([u for u in u_full], dim=0).std(),
        "v_std": torch.cat([v for v in v_full], dim=0).std(),
        "p_std": torch.cat([p for p in p_full], dim=0).std(),     # On ajoute la pression du bord
    }
    X_full = torch.zeros((0, 3))
    U_full = torch.zeros((0, 3))
    x_norm_full.append((x_full[0] - mean_std["x_mean"]) / mean_std["x_std"])
    y_norm_full.append((y_full[0] - mean_std["y_mean"]) / mean_std["y_std"])
    t_norm_full.append((t_full[0] - mean_std["t_mean"]) / mean_std["t_std"])
    p_norm_full.append((p_full[0] - mean_std["p_mean"]) / mean_std["p_std"])
    u_norm_full.append((u_full[0] - mean_std["u_mean"]) / mean_std["u_std"])
    v_norm_full.append((v_full[0] - mean_std["v_mean"]) / mean_std["v_std"])
    X_full = torch.cat(
        (
            X_full,
            torch.stack(
                (
                    x_norm_full[-1],
                    y_norm_full[-1],
                    t_norm_full[-1],
                ),
                dim=1,
            ),
        )
    )
    U_full = torch.cat(
        (
            U_full,
            torch.stack((u_norm_full[-1], v_norm_full[-1], p_norm_full[-1]), dim=1),
        )
    )
    # time_unique = X_full[:, 2].unique()
    # time_perfect = torch.linspace(time_unique.min(), time_unique.max(), hyper_param['nb_timestep'])
    # closest_time = torch.zeros((hyper_param['nb_timestep']))
    # for k, t_ in enumerate(time_perfect):
    #     diff_time = torch.abs(time_unique-t_) 
    #     closest_time[k] = time_unique[torch.argmin(diff_time)]
    # masque = torch.isin(X_full[:, 2], closest_time)
    # X_full_time = X_full[masque]
    # U_full_time = U_full[masque]
    return X_full, U_full, mean_std


def init_model(f, hyper_param, device, folder_result):
    model = GNN(hyper_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper_param["lr_init"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyper_param["gamma_scheduler"]
    )
    loss = nn.MSELoss()
    # On regarde si notre modèle n'existe pas déjà
    if Path(folder_result + "/model_weights.pth").exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result + "/model_weights.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # weights = checkpoint["weights"]
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        # csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            # "data": list(csv_train["data"]),
            # "pde": list(csv_train["pde"]),
            # "border": list(csv_train["border"]),
        }

        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": []}
    return model, optimizer, scheduler, loss, train_loss


if __name__ == "__main__":
    write_csv([[1, 2, 3], [4, 5, 6]], "ready_cluster/piche/test.csv")
