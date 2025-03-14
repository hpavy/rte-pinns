import torch
from run import RunSimulation
from constants import DICT_Y0, DICT_CASE, PARAM_ADIM, M

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le code se lance sur {device}")


folder_result_name = "4_with_bruit"  # name of the result folder

# On utilise hyper_param_init uniquement si c'est un nouveau modèle

hyper_param_init = {
    "num": 1,
    "case": 2,
    "nb_epochs": 10000,
    "save_rate": 300,
    "batch_size": 10,
    "lr_init": 1e-3,
    "gamma_scheduler": 0.999,
    "x_min": -0.06,
    "x_max": 0.06,
    "y_min": -0.06,
    "y_max": 0.06,
    "t_min": 6.5,
    'nb_hidden': 2,
    'dim_latent': 32,
    'nb_processors': 10,
    'nb_neighbours': 6
}

hyper_param_init['H'] = [DICT_CASE[str(hyper_param_init['case'])]]
hyper_param_init['ya0'] = [DICT_Y0[str(hyper_param_init['num'])]]
hyper_param_init['file'] = [
    f"model_{hyper_param_init['num']}_case_{hyper_param_init['case']}.csv"
    ]
hyper_param_init['m'] = M

simu = RunSimulation(hyper_param_init, folder_result_name, PARAM_ADIM)

simu.run()
