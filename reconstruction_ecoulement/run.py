from geometry import RectangleWithoutCylinder
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
        # Creation du dossier de result
        Path(self.folder_result).mkdir(parents=True, exist_ok=True)
        if not Path(self.folder_result + "/hyper_param.json").exists():
            with open(self.folder_result + "/hyper_param.json", "w") as file:
                json.dump(self.hyper_param, file, indent=4)
            self.hyper_param = self.hyper_param

        else:
            with open(self.folder_result + "/hyper_param.json", "r") as file:
                self.hyper_param = json.load(file)

        # Chargement des données
        (
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
        ) = charge_data(self.hyper_param, self.param_adim)
        X_train.requires_grad_()
        U_train.requires_grad_()
        X_border_train.requires_grad_()
        U_border_train.requires_grad_()

        # On écrit mean_std
        mean_std_json = dict()
        for key in mean_std:
            mean_std_json[key] = mean_std[key].item()
        with open(self.folder_result + "/mean_std.json", "w") as file:
            json.dump(mean_std_json, file, indent=4)

        # On plot les print dans un fichier texte
        with open(self.folder_result + "/print.txt", "a") as f:
            model, optimizer, scheduler, loss, train_loss, test_loss, weights = (
                init_model(f, self.hyper_param, self.device, self.folder_result)
            )
            
            if self.hyper_param["dynamic_weights"]:
                weight_data = weights["weight_data"]
                weight_pde = weights["weight_pde"]
                weight_border = weights["weight_border"]
            else:
                weight_data = self.hyper_param["weight_data"]
                weight_pde = self.hyper_param["weight_pde"]
                weight_border = self.hyper_param["weight_border"]
                # Data loading

            # On entraine le modèle
            train(
                nb_epoch=1000,
                train_loss=train_loss,
                test_loss=test_loss,
                weight_data_init=weight_data,
                weight_pde_init=weight_pde,
                weight_border_init=weight_border,
                dynamic_weights=self.hyper_param["dynamic_weights"],
                lr_weights=self.hyper_param["lr_weights"],
                model=model,
                loss=loss,
                optimizer=optimizer,
                X_train=X_train,
                U_train=U_train,
                X_pde=X_pde,
                X_test_pde=X_test_pde,
                X_test_data=X_test_data,
                U_test_data=U_test_data,
                Re=self.hyper_param["Re"],
                time_start=self.time_start,
                f=f,
                folder_result=self.folder_result,
                save_rate=self.hyper_param["save_rate"],
                batch_size=self.hyper_param["batch_size"],
                scheduler=scheduler,
                X_border_train=X_border_train,
                X_border_test=X_border_test,
                U_border_train=U_border_train,
                U_border_test=U_border_test,
                mean_std=mean_std,
                param_adim=self.param_adim,
                nb_simu=len(self.hyper_param["file"]),
                force_inertie_bool=self.hyper_param["force_inertie_bool"],
                u_border=self.hyper_param['u_border'],
                v_border=self.hyper_param['v_border'],
                p_border=self.hyper_param['p_border'],
                with_pinns=self.hyper_param['with_pinns']
            )
