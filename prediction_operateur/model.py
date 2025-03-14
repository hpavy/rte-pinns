import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, nb_entry, nb_neurons, nb_layers, nb_branches):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(nb_entry, nb_neurons)])
        self.hiden_layers = nn.ModuleList(
            [nn.Linear(nb_neurons, nb_neurons) for _ in range(nb_layers - 1)]
        )
        self.final_layer = nn.ModuleList([nn.Linear(nb_neurons, nb_branches)])
        self.layers = self.init_layer + self.hiden_layers + self.final_layer
        self.initial_param()

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            if k != len(self.layers) - 1:
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class DeepONetSolo(nn.Module):
    def __init__(
        self,
        nb_entry_branch,
        nb_entry_trunk,
        trunk_width,
        trunk_depth,
        branch_width,
        branch_depth,
        nb_branches,
    ):
        super().__init__()
        self.trunk = MLP(nb_entry_trunk, trunk_width, trunk_depth, nb_branches)
        self.branch = MLP(nb_entry_branch, branch_width, branch_depth, nb_branches)
        self.trunk.initial_param()
        self.branch.initial_param()
        self.nb_branches = nb_branches
        self.final_bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))])

    def forward(self, x_branch, x_trunk):
        x_loc = torch.relu(self.trunk(x_trunk))
        x_func = self.branch(x_branch)
        product_branch = x_loc * x_func
        return torch.sum(product_branch, dim=1, keepdim=True) + self.final_bias[0]


class DeepONet(nn.Module):
    def __init__(self, hyper_param):
        nb_exit = hyper_param["nb_exit"]
        nb_entry_branch = hyper_param["nb_entry_branch"]
        nb_entry_trunk = hyper_param["nb_entry_trunk"]
        trunk_width = hyper_param["trunk_width"]
        trunk_depth = hyper_param["trunk_depth"]
        branch_width = hyper_param["branch_width"]
        branch_depth = hyper_param["branch_depth"]
        nb_branches = hyper_param["nb_branches"]
        super().__init__()
        self.list_op = torch.nn.ParameterList(
            [
                DeepONetSolo(
                    nb_entry_branch,
                    nb_entry_trunk,
                    trunk_width,
                    trunk_depth,
                    branch_width,
                    branch_depth,
                    nb_branches,
                )
                for _ in range(nb_exit)
            ]
        )

    def forward(self, x_branch, x_trunk):
        result = [o_net(x_branch, x_trunk) for o_net in self.list_op]
        return torch.stack([y.flatten() for y in result], dim=1)


if __name__ == "__main__":
    piche = DeepONet(
        nb_entry_branch=1,
        nb_entry_trunk=3,
        trunk_width=64,
        trunk_depth=6,
        branch_width=64,
        branch_depth=6,
        nb_branches=20,
        nb_exit=3,
    )
    nombre_parametres = sum(p.numel() for p in piche.parameters() if p.requires_grad)
    print(nombre_parametres)
