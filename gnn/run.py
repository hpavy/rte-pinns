import torch
from utils import charge_data, init_model
from train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np
import json


class RunSimulation:
    def __init__(self, hyper_param, folder_result_name, param_adim):
        self.hyper_param = hyper_param
        self.time_start = time.time()
        self.folder_result_name = folder_result_name
        self.folder_result = "results/" + folder_result_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_adim = param_adim
        self.nb_simu = len(self.hyper_param["file"])

    def run(self):
        # Charging the model
        # Creation du dossier de result
        Path(self.folder_result).mkdir(parents=True, exist_ok=True)
        if not Path(self.folder_result + "/hyper_param.json").exists():
            with open(self.folder_result + "/hyper_param.json", "w") as file:
                json.dump(self.hyper_param, file, indent=4)
            self.hyper_param = self.hyper_param

        else:
            with open(self.folder_result + "/hyper_param.json", "r") as file:
                self.hyper_param = json.load(file)
        (
            X_full,
            U_full,
            mean_std,
        ) = charge_data(self.hyper_param, self.param_adim)
        mean_std_json = dict()
        for key in mean_std:
            mean_std_json[key] = mean_std[key].item()
        with open(self.folder_result + "/mean_std.json", "w") as file:
            json.dump(mean_std_json, file, indent=4)

        # Initialiser le modèle

        # On plot les print dans un fichier texte
        with open(self.folder_result + "/print.txt", "a") as f:
            model, optimizer, scheduler, loss, train_loss = (
                init_model(f, self.hyper_param, self.device, self.folder_result)
            )
            # On entraine le modèle
            train(
                nb_epoch=self.hyper_param['nb_epochs'],
                train_loss=train_loss,
                model=model,
                loss=loss,
                optimizer=optimizer,
                time_start=self.time_start,
                f=f,
                folder_result=self.folder_result,
                save_rate=self.hyper_param["save_rate"],
                batch_size=self.hyper_param["batch_size"],
                scheduler=scheduler,
                X_full=X_full,
                U_full=U_full,
                nb_neighbours=self.hyper_param['nb_neighbours']
            )
