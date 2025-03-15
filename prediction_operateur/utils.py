# Les fonctions utiles ici
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from model import DeepONet
import torch
import time
from geometry import RectangleWithoutCylinder
import numpy as np


def write_csv(data, path, file_name):
    dossier = Path(path)
    df = pd.DataFrame(data)
    # Créer le dossier si il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    df.to_csv(path + file_name)


def read_csv(path):
    return pd.read_csv(path)


def find_x_branch(ya0, H, m, nb_branch, t_min, t_max):
    f = 0.5 * (H / m) ** 0.5
    w_0 = 2 * torch.pi * f
    time_interval = torch.linspace(t_min, t_max, nb_branch)
    return ya0 * (w_0**2) * torch.cos(w_0 * time_interval)


def charge_data(hyperparam, param_adim):
    """
    Charge the data of X_full, U_full with every points
    And X_train, U_train with less points
    """
    # La data
    # On adimensionne la data (normalisation Z)
    time_start_charge = time.time()
    nb_simu = len(hyperparam["file"])

    # On charge les fichiers
    x_full, y_full, t_full, x_branch_full = [], [], [], []
    u_full, v_full, p_full = [], [], []
    x_border, y_border, t_border, x_branch_border = [], [], [], []
    u_border, v_border, p_border = [], [], []
    x_norm_full, y_norm_full, t_norm_full, x_branch_norm_full = (
        [],
        [],
        [],
        [],
    )
    u_norm_full, v_norm_full, p_norm_full = [], [], []
    x_norm_border, y_norm_border, t_norm_border, x_branch_norm_border = (
        [],
        [],
        [],
        [],
    )
    u_norm_border, v_norm_border, p_norm_border = [], [], []
    H_numpy = np.array(hyperparam["H"])
    f_numpy = 0.5 * (H_numpy / hyperparam["m"]) ** 0.5
    f = np.min(f_numpy)  # La plus petite période
    t_max = hyperparam["t_min"] + hyperparam["nb_period"] / f
    for k in range(nb_simu):
        # On charge le csv
        df = pd.read_csv("data/" + hyperparam["file"][k])
        df_modified = df.loc[
            (df["Points:0"] >= hyperparam["x_min"])
            & (df["Points:0"] <= hyperparam["x_max"])
            & (df["Points:1"] >= hyperparam["y_min"])
            & (df["Points:1"] <= hyperparam["y_max"])
            & (df["Time"] > hyperparam["t_min"])
            & (df["Time"] < t_max)
            & (df["Points:2"] == 0.0)
            & (df["Points:0"] ** 2 + df["Points:1"] ** 2 > hyperparam["r_min"] ** 2),
            :,
        ].copy()
        df_modified.loc[:, "ya0"] = hyperparam["ya0"][k]
        df_modified.loc[:, "w0"] = (
            torch.pi * (hyperparam["H"][k] / hyperparam["m"]) ** 0.5
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
        t_max_plot = hyperparam["t_min"] + 1 / f_flow
        time_without_modulo = df_modified["Time"].to_numpy() - hyperparam["t_min"]
        time_with_modulo = hyperparam["t_min"] + time_without_modulo % (1 / f_flow)
        # Pour avoir plus de pas de temps sur une période, on ramène toutes les données sur la même période
        t_full.append(
            torch.tensor(time_with_modulo, dtype=torch.float32)
            / (param_adim["L"] / param_adim["V"])
        )
        x_branch_full.append(
            find_x_branch(
                hyperparam["ya0"][k],
                H=hyperparam["H"][k],
                m=hyperparam["m"],
                nb_branch=hyperparam["nb_entry_branch"],
                t_min=hyperparam["t_min"],
                t_max=t_max_plot,
            )
            .reshape(-1, 1)
            .repeat(1, x_full[-1].shape[0])
            .T
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

        df_border = pd.read_csv("data/" + hyperparam["file"][k][:-4] + "_border.csv")
        df_modified_border = df_border.loc[
            (df_border["Time"] > hyperparam["t_min"])
            & (df_border["Time"] < t_max)
            & (df_border["Points:2"] == 0.0),
            :,
        ].copy()

        df_modified_border.loc[:, "ya0"] = hyperparam["ya0"][k]
        df_modified_border.loc[:, "w0"] = (
            torch.pi * (hyperparam["H"][k] / hyperparam["m"]) ** 0.5
        )

        df_modified_border.loc[:, "theta"] = np.arctan2(
            df_modified_border["Points:1"], df_modified_border["Points:0"]
        )

        # on ne garde que ceux loin de pi et 0
        df_modified_border = df_modified_border.loc[
            (np.abs(df_modified_border["theta"]) > hyperparam["theta_border_min"])
            & (
                np.abs(np.pi - np.abs(df_modified_border["theta"]))
                > hyperparam["theta_border_min"]
            )
        ]
        df_modified_border.loc[:, "Pressure"] = -df_modified_border[
            "Stress:1"
        ] / np.sin(df_modified_border["theta"])

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
        time_without_modulo = (
            df_modified_border["Time"].to_numpy() - hyperparam["t_min"]
        )
        time_with_modulo = hyperparam["t_min"] + time_without_modulo % (1 / f_flow)
        t_border.append(
            torch.tensor(time_with_modulo, dtype=torch.float32)
            / (param_adim["L"] / param_adim["V"])
        )
        x_branch_border.append(
            find_x_branch(
                hyperparam["ya0"][k],
                H=hyperparam["H"][k],
                m=hyperparam["m"],
                nb_branch=hyperparam["nb_entry_branch"],
                t_min=hyperparam["t_min"],
                t_max=t_max_plot,
            )
            .reshape(-1, 1)
            .repeat(1, x_border[-1].shape[0])
            .T
        )
        u_border.append(0.0 * torch.zeros(t_border[k].shape[0], dtype=torch.float32))
        v_border.append(0.0 * torch.zeros(t_border[k].shape[0], dtype=torch.float32))
        p_border.append(
            torch.tensor(df_modified_border["Pressure"].to_numpy(), dtype=torch.float32)
            / ((param_adim["V"] ** 2) * param_adim["rho"])
        )
        print(f"fichier n°{k} chargé")

    # les valeurs pour renormaliser ou dénormaliser
    mean_std = {
        "u_mean": torch.cat([u for u in u_full], dim=0).mean(),
        "v_mean": torch.cat([v for v in v_full], dim=0).mean(),
        "p_mean": torch.cat(
            [torch.cat((p, p_), dim=0) for p, p_ in zip(p_full, p_border)], dim=0
        ).mean(),  # On ajoute la pression du bord
        "x_mean": torch.cat([x for x in x_full], dim=0).mean(),
        "y_mean": torch.cat([y for y in y_full], dim=0).mean(),
        "t_mean": torch.cat([t for t in t_full], dim=0).mean(),
        "x_std": torch.cat([x for x in x_full], dim=0).std(),
        "y_std": torch.cat([y for y in y_full], dim=0).std(),
        "t_std": torch.cat([t for t in t_full], dim=0).std(),
        "u_std": torch.cat([u for u in u_full], dim=0).std(),
        "v_std": torch.cat([v for v in v_full], dim=0).std(),
        "p_std": torch.cat(
            [torch.cat((p, p_), dim=0) for p, p_ in zip(p_full, p_border)], dim=0
        ).std(),  # On ajoute la pression du bord
        "x_branch_std": torch.cat([x_b for x_b in x_branch_full], dim=0).std(dim=0),
        "x_branch_mean": torch.cat([x_b for x_b in x_branch_full], dim=0).mean(dim=0),
    }

    X_trunk_full = torch.zeros((0, 3))
    U_full = torch.zeros((0, 3))
    X_trunk_border = torch.zeros((0, 3))
    U_border = torch.zeros((0, 3))
    for k in range(nb_simu):
        # Normalisation Z
        x_norm_full.append((x_full[k] - mean_std["x_mean"]) / mean_std["x_std"])
        y_norm_full.append((y_full[k] - mean_std["y_mean"]) / mean_std["y_std"])
        t_norm_full.append((t_full[k] - mean_std["t_mean"]) / mean_std["t_std"])
        x_branch_norm_full.append(
            (x_branch_full[k] - mean_std["x_branch_mean"]) / mean_std["x_branch_std"]
        )
        p_norm_full.append((p_full[k] - mean_std["p_mean"]) / mean_std["p_std"])
        u_norm_full.append((u_full[k] - mean_std["u_mean"]) / mean_std["u_std"])
        v_norm_full.append((v_full[k] - mean_std["v_mean"]) / mean_std["v_std"])
        X_trunk_full = torch.cat(
            (
                X_trunk_full,
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

        x_norm_border.append((x_border[k] - mean_std["x_mean"]) / mean_std["x_std"])
        y_norm_border.append((y_border[k] - mean_std["y_mean"]) / mean_std["y_std"])
        t_norm_border.append((t_border[k] - mean_std["t_mean"]) / mean_std["t_std"])
        x_branch_norm_border.append(
            (x_branch_border[k] - mean_std["x_branch_mean"]) / mean_std["x_branch_std"]
        )
        p_norm_border.append((p_border[k] - mean_std["p_mean"]) / mean_std["p_std"])
        u_norm_border.append((u_border[k] - mean_std["u_mean"]) / mean_std["u_std"])
        v_norm_border.append((v_border[k] - mean_std["v_mean"]) / mean_std["v_std"])
        X_trunk_border = torch.cat(
            (
                X_trunk_border,
                torch.stack(
                    (
                        x_norm_border[-1],
                        y_norm_border[-1],
                        t_norm_border[-1],
                    ),
                    dim=1,
                ),
            )
        )
        U_border = torch.cat(
            (
                U_border,
                torch.stack(
                    (u_norm_border[-1], v_norm_border[-1], p_norm_border[-1]), dim=1
                ),
            )
        )

    # Les valeurs qu'on va essayer de fitter
    X_branch_full = torch.cat([x_b for x_b in x_branch_norm_full])
    X_branch_border = torch.cat([x_b for x_b in x_branch_norm_border])
    X_trunk_train = torch.zeros((0, 3))
    X_branch_train = torch.zeros((0, 100))
    U_train = torch.zeros((0, 3))
    print("Starting X_train")
    print("Starting X_train")
    for nb in range(len(x_full)):
        print(f"Simu n°{nb}/{len(hyperparam['ya0'])}")
        print(f"Time:{(time.time()-time_start_charge):.3f}")
        for time_ in torch.unique(t_norm_full[nb]):
            masque = t_norm_full[nb] == time_
            indices = torch.randperm(x_norm_full[nb][masque].size(0))[
                : hyperparam["nb_points"]
            ]
            new_x = torch.stack(
                (
                    x_norm_full[nb][masque][indices],
                    y_norm_full[nb][masque][indices],
                    torch.ones(hyperparam["nb_points"]) * time_,
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
            X_trunk_train = torch.cat((X_trunk_train, new_x))
            U_train = torch.cat((U_train, new_y))
            X_branch_train = torch.cat(
                (X_branch_train, x_branch_norm_full[nb][indices, :])
            )
    indices = torch.randperm(X_trunk_train.size(0))
    X_trunk_train = X_trunk_train[indices]
    U_train = U_train[indices]
    X_branch_train = X_branch_train[indices]
    print("X_train OK")

    # les points du bord
    # border test et train

    indices = torch.randperm(X_branch_border.size(0))
    X_branch_border_train = X_branch_border[indices][
        : int(0.8 * X_branch_border.shape[0])
    ]
    X_trunk_border_train = X_trunk_border[indices][: int(0.8 * X_trunk_border.shape[0])]
    U_border_train = U_border[indices][: int(0.8 * U_border.shape[0])]
    X_branch_border_test = X_branch_border[indices][
        int(0.8 * X_branch_border.shape[0]) :
    ]
    X_trunk_border_test = X_trunk_border[indices][int(0.8 * X_trunk_border.shape[0]) :]
    U_border_test = U_border[indices][int(0.8 * U_border.shape[0]) :]

    points_coloc_test = np.random.choice(
        len(X_branch_full), hyperparam["n_data_test"], replace=False
    )
    X_branch_test_data = X_branch_full[points_coloc_test]
    X_trunk_test_data = X_trunk_full[points_coloc_test]
    U_test_data = U_full[points_coloc_test]
    return (
        X_branch_train,
        X_trunk_train,
        U_train,
        X_branch_test_data,
        X_trunk_test_data,
        U_test_data,
        U_train,
        X_branch_full,
        X_trunk_full,
        U_full,
        X_branch_border_train,
        X_trunk_border_train,
        X_branch_border_test,
        X_trunk_border_test,
        U_border_train,
        U_border_test,
        mean_std,
    )


def init_model(f, hyperparam, device, folder_result):
    model = DeepONet(hyperparam).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparam["lr_init"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyperparam["gamma_scheduler"]
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
            "border": list(csv_train["border"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "border": list(csv_test["border"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "border": []}
        test_loss = {"total": [], "data": [], "border": []}
        weights = {
            "weight_data": hyperparam["weight_data"],
            "weight_border": hyperparam["weight_border"],
        }
    return model, optimizer, scheduler, loss, train_loss, test_loss, weights


if __name__ == "__main__":
    write_csv([[1, 2, 3], [4, 5, 6]], "ready_cluster/piche/test.csv")
