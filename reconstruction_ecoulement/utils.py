# Les fonctions utiles ici
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from model import PINNs
import torch
import time
from geometry import RectangleWithoutCylinder
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


class CustomDataset(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


def charge_data(hyper_param, param_adim):
    """
    Charge the data of X_full, U_full with every points
    And X_train, U_train with less points
    """
    # La data
    # On adimensionne la data
    time_start_charge = time.time()
    nb_simu = len(hyper_param["file"])
    x_full, y_full, t_full, ya0_full, w0_full = [], [], [], [], []
    u_full, v_full, p_full = [], [], []
    x_border, y_border, t_border, ya0_border, w0_border = [], [], [], [], []
    u_border, v_border, p_border = [], [], []
    x_norm_full, y_norm_full, t_norm_full, ya0_norm_full, w0_norm_full = (
        [],
        [],
        [],
        [],
        [],
    )
    u_norm_full, v_norm_full, p_norm_full = [], [], []
    x_norm_border, y_norm_border, t_norm_border, ya0_norm_border, w0_norm_border = (
        [],
        [],
        [],
        [],
        [],
    )
    u_norm_border, v_norm_border, p_norm_border = [], [], []
    H_numpy = np.array(hyper_param["H"])
    f_numpy = 0.5 * (H_numpy / hyper_param["m"]) ** 0.5
    f = np.min(f_numpy)
    t_max = hyper_param["t_min"] + hyper_param["nb_period"] / f
    for k in range(nb_simu):
        df = pd.read_csv("data/" + hyper_param["file"][k])
        df_modified = df.loc[
            (df["Points:0"] >= hyper_param["x_min"])
            & (df["Points:0"] <= hyper_param["x_max"])
            & (df["Points:1"] >= hyper_param["y_min"])
            & (df["Points:1"] <= hyper_param["y_max"])
            & (df["Time"] > hyper_param["t_min"])
            & (df["Time"] < t_max)
            & (df["Points:2"] == 0.0)
            & (df["Points:0"] ** 2 + df["Points:1"] ** 2 > hyper_param['r_min'] ** 2),
            :,
        ].copy()
        df_modified.loc[:, "ya0"] = hyper_param["ya0"][k]
        df_modified.loc[:, "w0"] = (
            torch.pi * (hyper_param["H"][k] / hyper_param["m"]) ** 0.5
        )

        # Adimensionnement
        x_full.append(
            torch.tensor(df_modified["Points:0"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        y_full.append(
            torch.tensor(df_modified["Points:1"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        f_flow = f_numpy[k]
        time_without_modulo = df_modified["Time"].to_numpy() - hyper_param['t_min']
        time_with_modulo = hyper_param['t_min'] + time_without_modulo % (1/f_flow)
        t_full.append(
            torch.tensor(time_with_modulo, dtype=torch.float32)
            / (param_adim["L"] / param_adim["V"])
        )
        ya0_full.append(
            torch.tensor(df_modified["ya0"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        w0_full.append(
            torch.tensor(df_modified["w0"].to_numpy(), dtype=torch.float32)
            / (param_adim["V"] / param_adim["L"])
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
        ### Le border

        df_border = pd.read_csv("data/" + hyper_param["file"][k][:-4]+'_border.csv')
        df_modified_border = df_border.loc[
            (df_border["Time"] > hyper_param["t_min"])
            & (df_border["Time"] < t_max)
            & (df_border["Points:2"] == 0.0),
            :,
        ].copy()

        df_modified_border.loc[:, "ya0"] = hyper_param["ya0"][k]
        df_modified_border.loc[:, "w0"] = (
            torch.pi * (hyper_param["H"][k] / hyper_param["m"]) ** 0.5
        )

        df_modified_border.loc[:, "theta"] = np.arctan2(df_modified_border['Points:1'], df_modified_border['Points:0'])

        # on ne garde que ceux loin de pi et 0
        df_modified_border = df_modified_border.loc[
            (np.abs(df_modified_border["theta"]) > hyper_param['theta_border_min'])
            & (np.abs(np.pi - np.abs(df_modified_border["theta"])) > hyper_param['theta_border_min'])
        ]
        df_modified_border.loc[:, 'Pressure'] = - df_modified_border['Stress:1'] / np.sin(df_modified_border["theta"])

        # Adimensionnement
        x_border.append(
            torch.tensor(df_modified_border["Points:0"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        y_border.append(
            torch.tensor(df_modified_border["Points:1"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        f_flow = f_numpy[k]
        time_without_modulo = df_modified_border["Time"].to_numpy() - hyper_param["t_min"]
        time_with_modulo = hyper_param["t_min"] + time_without_modulo % (1 / f_flow)
        t_border.append(
            torch.tensor(time_with_modulo, dtype=torch.float32)
            / (param_adim["L"] / param_adim["V"])
        )
        ya0_border.append(
            torch.tensor(df_modified_border["ya0"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        w0_border.append(
            torch.tensor(df_modified_border["w0"].to_numpy(), dtype=torch.float32)
            / (param_adim["V"] / param_adim["L"])
        )
        u_border.append(
            0. * torch.zeros(w0_border[k].shape[0], dtype=torch.float32)
        )
        v_border.append(
            0. * torch.zeros(w0_border[k].shape[0], dtype=torch.float32)
        )
        p_border.append(
            torch.tensor(df_modified_border["Pressure"].to_numpy(), dtype=torch.float32)
            / ((param_adim["V"] ** 2) * param_adim["rho"])
        )
        print(f"fichier n°{k} chargé")
        
    if nb_simu == 1:
        w0_std = torch.ones(1)
        ya0_std = torch.ones(1)
    else:
        w0_std = torch.cat([w0 for w0 in w0_full], dim=0).std()
        ya0_std = torch.cat([ya0 for ya0 in ya0_full], dim=0).std()

    # les valeurs pour renormaliser ou dénormaliser
    mean_std = {
        "u_mean": torch.cat([u for u in u_full], dim=0).mean(),
        "v_mean": torch.cat([v for v in v_full], dim=0).mean(),
        "p_mean": torch.cat([torch.cat((p, p_), dim=0) for p, p_ in zip(p_full, p_border)], dim=0).mean(),   # On ajoute la pression du bord
        "x_mean": torch.cat([x for x in x_full], dim=0).mean(),
        "y_mean": torch.cat([y for y in y_full], dim=0).mean(),
        "t_mean": torch.cat([t for t in t_full], dim=0).mean(),
        "x_std": torch.cat([x for x in x_full], dim=0).std(),
        "y_std": torch.cat([y for y in y_full], dim=0).std(),
        "t_std": torch.cat([t for t in t_full], dim=0).std(),
        "u_std": torch.cat([u for u in u_full], dim=0).std(),
        "v_std": torch.cat([v for v in v_full], dim=0).std(),
        "p_std": torch.cat([torch.cat((p, p_), dim=0) for p, p_ in zip(p_full, p_border)], dim=0).std(),     # On ajoute la pression du bord
        "ya0_mean": torch.cat([ya0 for ya0 in ya0_full], dim=0).mean(),
        "ya0_std": ya0_std,
        "w0_mean": torch.cat([w0 for w0 in w0_full], dim=0).mean(),
        "w0_std": w0_std,
    }

    X_full = torch.zeros((0, 5))
    U_full = torch.zeros((0, 3))
    X_border = torch.zeros((0, 5))
    U_border = torch.zeros((0, 3))
    for k in range(nb_simu):
        # Normalisation Z
        x_norm_full.append((x_full[k] - mean_std["x_mean"]) / mean_std["x_std"])
        y_norm_full.append((y_full[k] - mean_std["y_mean"]) / mean_std["y_std"])
        t_norm_full.append((t_full[k] - mean_std["t_mean"]) / mean_std["t_std"])
        ya0_norm_full.append((ya0_full[k] - mean_std["ya0_mean"]) / mean_std["ya0_std"])
        w0_norm_full.append((w0_full[k] - mean_std["w0_mean"]) / mean_std["w0_std"])
        p_norm_full.append((p_full[k] - mean_std["p_mean"]) / mean_std["p_std"])
        u_norm_full.append((u_full[k] - mean_std["u_mean"]) / mean_std["u_std"])
        v_norm_full.append((v_full[k] - mean_std["v_mean"]) / mean_std["v_std"])
        X_full = torch.cat(
            (
                X_full,
                torch.stack(
                    (
                        x_norm_full[-1],
                        y_norm_full[-1],
                        t_norm_full[-1],
                        ya0_norm_full[-1],
                        w0_norm_full[-1],
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

        x_norm_border.append((x_border[k] - mean_std["x_mean"]) / mean_std["x_std"])
        y_norm_border.append((y_border[k] - mean_std["y_mean"]) / mean_std["y_std"])
        t_norm_border.append((t_border[k] - mean_std["t_mean"]) / mean_std["t_std"])
        ya0_norm_border.append((ya0_border[k] - mean_std["ya0_mean"]) / mean_std["ya0_std"])
        w0_norm_border.append((w0_border[k] - mean_std["w0_mean"]) / mean_std["w0_std"])
        p_norm_border.append((p_border[k] - mean_std["p_mean"]) / mean_std["p_std"])
        u_norm_border.append((u_border[k] - mean_std["u_mean"]) / mean_std["u_std"])
        v_norm_border.append((v_border[k] - mean_std["v_mean"]) / mean_std["v_std"])
        X_border = torch.cat(
            (
                X_border,
                torch.stack(
                    (
                        x_norm_border[-1],
                        y_norm_border[-1],
                        t_norm_border[-1],
                        ya0_norm_border[-1],
                        w0_norm_border[-1],
                    ),
                    dim=1,
                ),
            )
        )
        U_border = torch.cat(
            (
                U_border,
                torch.stack((u_norm_border[-1], v_norm_border[-1], p_norm_border[-1]), dim=1),
            )
        )

    X_train = torch.zeros((0, 5))
    U_train = torch.zeros((0, 3))
    print("Starting X_train")
    for nb, ya0_ in enumerate(hyper_param["ya0"]):
        print(f"Simu n°{nb}/{len(hyper_param['ya0'])}")
        print(f"Time:{(time.time()-time_start_charge):.3f}")
        w_0 = w0_norm_full[nb][0].item()
        y_0 = ya0_norm_full[nb][0].item()
        for time_ in torch.unique(t_norm_full[nb]):
            masque = t_norm_full[nb] == time_
            indices = torch.randperm(x_norm_full[nb][masque].size(0))[
                : hyper_param["nb_points"]
            ]
            new_x = torch.stack(
                (
                    x_norm_full[nb][masque][indices],
                    y_norm_full[nb][masque][indices],
                    torch.ones(hyper_param["nb_points"]) * time_,
                    torch.ones(hyper_param["nb_points"]) * y_0,
                    torch.ones(hyper_param["nb_points"]) * w_0,
                ),
                dim=1,
            )
            new_y = torch.stack(
                (
                    u_norm_full[nb][masque][indices],
                    v_norm_full[nb][masque][indices],
                    p_norm_full[nb][masque][indices],
                ),
                dim=1,
            )
            X_train = torch.cat((X_train, new_x))
            U_train = torch.cat((U_train, new_y))
    indices = torch.randperm(X_train.size(0))
    X_train = X_train[indices]
    U_train = U_train[indices]
    print("X_train OK")

    # les points du bord
    # border test et train

    indices = torch.randperm(X_border.size(0))
    X_border_train = X_border[indices][:int(0.8*X_border.shape[0])]
    U_border_train = U_border[indices][:int(0.8*X_border.shape[0])]
    X_border_test = X_border[indices][int(0.8*X_border.shape[0]):]
    U_border_test = U_border[indices][int(0.8*X_border.shape[0]):]

    # On charge le pde
    # le domaine de résolution
    rectangle = RectangleWithoutCylinder(
        x_max=X_full[:, 0].max(),
        y_max=X_full[:, 1].max(),
        t_min=X_full[:, 2].min(),
        t_max=X_full[:, 2].max(),
        x_min=X_full[:, 0].min(),
        y_min=X_full[:, 1].min(),
        x_cyl=0,
        y_cyl=0,
        r_cyl=0.025 / 2,
        mean_std=mean_std,
        param_adim=param_adim,
    )

    X_pde = torch.empty((hyper_param["nb_points_pde"] * nb_simu, 5))
    for nb in range(len(ya0_norm_full)):
        X_pde_without_param = torch.concat(
            (
                rectangle.generate_lhs(hyper_param["nb_points_pde"]),
                ya0_norm_full[nb][0] * torch.ones(hyper_param["nb_points_pde"]).reshape(-1, 1),
                torch.ones(hyper_param["nb_points_pde"]).reshape(-1, 1)
                * w0_norm_full[nb][0],
            ),
            dim=1,
        )
        X_pde[
            nb * hyper_param["nb_points_pde"]: (nb + 1) * hyper_param["nb_points_pde"]
        ] = X_pde_without_param
    indices = torch.randperm(X_pde.size(0))
    X_pde = X_pde[indices, :].detach()
    print("X_pde OK")

    # Data test loading
    X_test_pde = torch.empty((hyper_param["n_pde_test"] * nb_simu, 5))
    for nb in range(len(ya0_norm_full)):
        X_test_pde_without_param = torch.concat(
            (
                rectangle.generate_lhs(hyper_param["n_pde_test"]),
                ya0_norm_full[nb][0] * torch.ones(hyper_param["n_pde_test"]).reshape(-1, 1),
                torch.ones(hyper_param["n_pde_test"]).reshape(-1, 1)
                * w0_norm_full[nb][0],
            ),
            dim=1,
        )

        X_test_pde[
            nb * hyper_param["n_pde_test"]: (nb + 1) * hyper_param["n_pde_test"]
        ] = X_test_pde_without_param
    indices = torch.randperm(X_test_pde.size(0))
    X_test_pde = X_test_pde[indices, :].detach()

    points_coloc_test = np.random.choice(
        len(X_full), hyper_param["n_data_test"], replace=False
    )
    X_test_data = X_full[points_coloc_test]
    U_test_data = U_full[points_coloc_test]
    return (
        X_train,
        U_train,
        X_full,
        U_full,
        X_border_train,
        X_border_test,
        U_border_train,
        U_border_test,
        mean_std,
        X_pde,
        X_test_pde,
        X_test_data,
        U_test_data,
    )


def init_model(f, hyper_param, device, folder_result):
    model = PINNs(hyper_param).to(device)
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
        weights = checkpoint["weights"]
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            "data": list(csv_train["data"]),
            "pde": list(csv_train["pde"]),
            "border": list(csv_train["border"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "pde": list(csv_test["pde"]),
            "border": list(csv_test["border"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "pde": [], "border": []}
        test_loss = {"total": [], "data": [], "pde": [], "border": []}
        weights = {
            "weight_data": hyper_param["weight_data"],
            "weight_pde": hyper_param["weight_pde"],
            "weight_border": hyper_param["weight_border"],
        }
    return model, optimizer, scheduler, loss, train_loss, test_loss, weights


if __name__ == "__main__":
    write_csv([[1, 2, 3], [4, 5, 6]], "ready_cluster/piche/test.csv")
