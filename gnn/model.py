import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, nb_hidden, dim_hidden):
        super().__init__()
        self.first = nn.Linear(dim_in, dim_hidden)
        self.hidden = nn.ModuleList([
            nn.Linear(dim_hidden, dim_hidden) for _ in range(nb_hidden-1)
        ])
        self.last = nn.Linear(dim_hidden, dim_out)
        self.initial_param()

    def forward(self, x):
        x = F.relu(self.first(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.last(x)  # pas d'activation sur la dernière
        return x

    def initial_param(self):
        for layer in self.hidden + [self.first, self.last]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class Encoder(nn.Module):
    def __init__(self, dim_latent, nb_hidden):
        super().__init__()
        self.nodes_encoder = MLP(dim_in=3, dim_out=dim_latent, nb_hidden=nb_hidden, dim_hidden=dim_latent)
        self.edges_attributes_encoder = MLP(dim_in=3, dim_out=dim_latent, nb_hidden=nb_hidden, dim_hidden=dim_latent)

    def forward(self, nodes, edges_attributes):
        """
        nodes : Nb_batch*Np*3
        edges_attributes : Np*Nv*3
        """
        nodes_encode = self.nodes_encoder(nodes)  # NB_batch*Np*Nl
        edges_attributes_encode = self.edges_attributes_encoder(edges_attributes)  # Np*Nv*Nl
        return nodes_encode, edges_attributes_encode


class Processor(nn.Module):
    def __init__(self, dim_latent, nb_hidden):
        super().__init__()
        self.mlp_edges = MLP(dim_in=3*dim_latent, dim_out=dim_latent, nb_hidden=nb_hidden, dim_hidden=dim_latent)
        self.mlp_nodes = MLP(dim_in=2*dim_latent, dim_out=dim_latent, nb_hidden=nb_hidden, dim_hidden=dim_latent)

    def forward(self, nodes, edges_attributes, edges_indices):
        """
        nodes : Nb_batch*Np*Nl
        edges_attributes : Np*Nv*Nl
        edges_indices : Np*Nv
        """
        nodes_neighbours = nodes[:, edges_indices]  # Nb_batch*Np*Nv*Nl
        concat_for_message = torch.concat(
            (
                nodes.unsqueeze(2).expand(-1, -1, edges_attributes.shape[1], -1),
                nodes_neighbours,
                edges_attributes.unsqueeze(0).expand(nodes.shape[0], -1, -1, -1)
                ), dim=-1
            )  # Nb_batch*Np*Nv*3Nl
        new_edges_attributes = self.mlp_edges(concat_for_message)  # Nb_batch*Np*Nv*Nl
        aggregate_message = torch.sum(new_edges_attributes, dim=2)  # Nb_batch*Np*Nl
        concat_for_nodes = torch.concat((nodes, aggregate_message), dim=2) # Nb_batch*Np*2Nl
        new_nodes = nodes + self.mlp_nodes(concat_for_nodes) # Nb_batch*Np*Nl + Nb_batch*Np*Nl = Nb_batch*Np*Nl
        return new_nodes, new_edges_attributes[0]


class Decoder(nn.Module):
    def __init__(self, dim_latent, nb_hidden):
        super().__init__()
        self.nodes_decoder = MLP(dim_in=dim_latent, dim_out=3, nb_hidden=nb_hidden, dim_hidden=dim_latent)

    def forward(self, nodes):
        """
        nodes : Np*Nl
        """
        nodes_decode = self.nodes_decoder(nodes)  # Np*3
        return nodes_decode


class GNN(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.encoder = Encoder(dim_latent=hyper_param['dim_latent'], nb_hidden=hyper_param['nb_hidden'])
        self.decoder = Decoder(dim_latent=hyper_param['dim_latent'], nb_hidden=hyper_param['nb_hidden'])
        self.processors = nn.ModuleList([
            Processor(dim_latent=hyper_param['dim_latent'], nb_hidden=hyper_param['nb_hidden']) for _ in range(hyper_param['nb_processors'])
            ])

    def forward(self, nodes, edges_attributes, edges_indices):
        nodes_encode, edges_attributes_encode = self.encoder(nodes, edges_attributes)
        for processor in self.processors:
            nodes_encode, edges_attributes_encode = processor(nodes_encode, edges_attributes_encode, edges_indices)
        nodes_decode = self.decoder(nodes_encode)
        return nodes_decode
