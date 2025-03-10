import torch
from run import RunSimulation
from constants import DICT_Y0, DICT_CASE, PARAM_ADIM, M

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le code se lance sur {device}")


folder_result_name = "piche"  # Le nom du fichier de résultats

# On utilise hyperparam_init uniquement si c'est un nouveau modèle
hyperparam_init = {
    "num": [1],
    "case": [2],
    "nb_epoch": 1000,  # le nombre d'epoch
    "save_rate": 10,
    "with_pinns": False,  # Si on utilise la loss
    "dynamic_weights": False,  # Est ce qu'on fait bouger de manière dynamique les poids
    "lr_weights": 0.1,  # la learning rate pour les poids si on les fait bouger 
    "weight_data": 0.33,  # le poids initial de la data
    "weight_pde": 0.33,
    "weight_border": 0.33,
    "batch_size": 10000,  # La taille d'un batch du pde
    "nb_points_pde": 1000000,  # Le nombre total de points pour le pde
    "Re": 100,
    "lr_init": 5e-4,
    "gamma_scheduler": 0.999,  # Pour la learning rate
    "nb_layers": 10,
    "nb_neurons": 64,
    "n_pde_test": 500,
    "n_data_test": 5000,
    "nb_points": 80,  # Nombre de points sur chaque pas de temps 
    # Les limites sur lesquelles on entraîne
    "x_min": -0.1,     
    "x_max": 0.1,
    "y_min": -0.1,
    "y_max": 0.1,
    "t_min": 6.5,
    "nb_period": 5,  # le nombre de période qu'on prend (on ramène tout sur une seule pour avoir plus de pas de temps)
    # "nb_period_plot": 2,
    "force_inertie_bool": False,  # Est ce qu'on prend la force d'inertie
    # Les conditons au bord
    "u_border": False,
    "v_border": False,
    "p_border": False,
    "r_min": 0.026/2,  # Le rayon minimal à partir du quel on prend les données
    'theta_border_min': 0.1,
    'is_res': True,  # Est ce qu'on utilise un res net
    'nb_blocks': 10,  # Pour ResNet
    'nb_layer_block': 3  # Pour ResNet
}

hyperparam_init['H'] = [DICT_CASE[str(k)] for k in hyperparam_init['case']]
hyperparam_init['ya0'] = [DICT_Y0[str(k)] for k in hyperparam_init['num']]
hyperparam_init['file'] = [
    f"model_{num_}_case_{case_}.csv" for num_, case_ in zip(hyperparam_init['num'], hyperparam_init['case'])
    ]
hyperparam_init['m'] = M

simu = RunSimulation(hyperparam_init, folder_result_name, PARAM_ADIM)  

simu.run()  # début de la simulation
