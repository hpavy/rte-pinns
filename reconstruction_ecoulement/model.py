import torch
import torch.nn as nn
from torch.autograd import grad


def pde(
    U,
    input,
    Re,
    x_std,
    y_std,
    u_mean,
    v_mean,
    p_std,
    t_std,
    t_mean,
    u_std,
    v_std,
    L_adim,
    V_adim,
    ya0_mean,
    ya0_std,
    w0_mean,
    w0_std,
    force_inertie_bool,
):
    # je sais qu'il fonctionne bien ! Il a été vérifié
    """Calcul la pde

    Args:
        U (_type_): u,v,p calcullés par le NN
        input (_type_): l'input (x,y,t)
    """
    u = U[:, 0].reshape(-1, 1)
    v = U[:, 1].reshape(-1, 1)
    p = U[:, 2].reshape(-1, 1)

    u_X = grad(
        outputs=u,
        inputs=input,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    v_X = grad(
        outputs=v,
        inputs=input,
        grad_outputs=torch.ones_like(v),
        retain_graph=True,
        create_graph=True,
    )[0]
    p_X = grad(
        outputs=p,
        inputs=input,
        grad_outputs=torch.ones_like(p),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_x = u_X[:, 0].reshape(-1, 1)
    u_y = u_X[:, 1].reshape(-1, 1)
    u_t = u_X[:, 2].reshape(-1, 1)
    v_x = v_X[:, 0].reshape(-1, 1)
    v_y = v_X[:, 1].reshape(-1, 1)
    v_t = v_X[:, 2].reshape(-1, 1)
    p_x = p_X[:, 0].reshape(-1, 1)
    p_y = p_X[:, 1].reshape(-1, 1)

    # Dans les prochaines lignes on peut améliorer le code (on fait des calculs inutiles)
    u_xx = grad(
        outputs=u_x, inputs=input, grad_outputs=torch.ones_like(u_x), retain_graph=True
    )[0][:, 0].reshape(-1, 1)
    u_yy = grad(
        outputs=u_y, inputs=input, grad_outputs=torch.ones_like(u_y), retain_graph=True
    )[0][:, 1].reshape(-1, 1)
    v_xx = grad(
        outputs=v_x, inputs=input, grad_outputs=torch.ones_like(v_x), retain_graph=True
    )[0][:, 0].reshape(-1, 1)
    v_yy = grad(
        outputs=v_y, inputs=input, grad_outputs=torch.ones_like(v_y), retain_graph=True
    )[0][:, 1].reshape(-1, 1)

    equ_1 = (
        (u_std / t_std) * u_t
        + (u * u_std + u_mean) * (u_std / x_std) * u_x
        + (v * v_std + v_mean) * (u_std / y_std) * u_y
        + (p_std / x_std) * p_x
        - (1 / Re) * ((u_std / (x_std**2)) * u_xx + (u_std / (y_std**2)) * u_yy)
    )
    if force_inertie_bool:
        force_inertie_ = - (
            (input[:, 3] * ya0_std + ya0_mean) * ((input[:, 4] * w0_std + w0_mean) ** 2)
            * torch.cos((input[:, 4] * w0_std + w0_mean) * (t_std * input[:, 2] + t_mean))
        )
    else:
        force_inertie_ = 0
    equ_2 = (
        (v_std / t_std) * v_t
        + (u * u_std + u_mean) * (v_std / x_std) * v_x
        + (v * v_std + v_mean) * (v_std / y_std) * v_y
        + (p_std / y_std) * p_y
        - (1 / Re) * ((v_std / (x_std**2)) * v_xx + (v_std / (y_std**2)) * v_yy)
        + force_inertie_
    )
    equ_3 = (u_std / x_std) * u_x + (v_std / y_std) * v_y
    return equ_1, equ_2, equ_3, force_inertie_, torch.mean((input[:, 4] * w0_std + w0_mean))*(V_adim/L_adim)


# Le NN


class PINNs(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(5, hyper_param["nb_neurons"])])
        self.hiden_layers = nn.ModuleList(
            [
                nn.Linear(hyper_param["nb_neurons"], hyper_param["nb_neurons"])
                for _ in range(hyper_param["nb_layers"] - 1)
            ]
        )
        self.final_layer = nn.ModuleList([nn.Linear(hyper_param["nb_neurons"], 3)])
        self.layers = self.init_layer + self.hiden_layers + self.final_layer
        self.initial_param()

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            if k != len(self.layers) - 1:
                x = torch.tanh(layer(x))
            else:
                x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


if __name__ == "__main__":
    hyper_param = {"nb_layers": 12, "nb_neurons": 64}
    piche = PINNs(hyper_param)
    nombre_parametres = sum(p.numel() for p in piche.parameters() if p.requires_grad)
    print(nombre_parametres)
