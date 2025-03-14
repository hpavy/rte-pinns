import torch
import time
from utils import write_csv
from pathlib import Path
from torch.utils.data import DataLoader
from utils import GraphDataset


def train(
    nb_epoch,
    train_loss,
    model,
    loss,
    optimizer,
    time_start,
    f,
    folder_result,
    save_rate,
    batch_size,
    scheduler,
    U_full,
    X_full,
    nb_neighbours
):
    print(f"Il y a {sum(p.numel() for p in model.parameters() if p.requires_grad)} parametres")
    print(f"Il y a {sum(p.numel() for p in model.parameters() if p.requires_grad)} parametres", file=f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_it_tot = nb_epoch + len(train_loss["total"])
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}"
        + "\n--------------------------"
    )
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}\n------------"
        + "--------------",
        file=f,
    )

    nb_batches = 10
    dataset = GraphDataset(X_full, U_full, nb_neighbours)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    edge_neighbours = dataset.edge_neighbours.to(device)
    edge_attributes = dataset.edge_attributes.to(device)
    delta_t = dataset.delta_t

    for epoch in range(len(train_loss["total"]), nb_it_tot):
        time_start_batch = time.time()
        total_batch = 0.0
        model.train()  # on dit qu'on va entrainer (on a le dropout)
        for i, batch in enumerate(dataloader):
            if i >= nb_batches:
                break

            # Supposons que batch contient (U, U_n)
            U, U_n = batch
            U += 0.05 * torch.randn_like(U)  # le bruit gaussien
            U = U.requires_grad_(True).to(device)
            U_n = U_n.to(device)

            pred_U_n = model(U, edge_attributes, edge_neighbours) * delta_t + U
            loss_data = loss(pred_U_n, U_n)

            # Backpropagation
            loss_data.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                total_batch += loss_data.item()
        scheduler.step()

        # Weights
        model.eval()
        with torch.no_grad():
            total_batch /= nb_batches
            train_loss["total"].append(total_batch)

        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}",
            file=f,
        )

        print(f"time: {time.time()-time_start:.0f}s")
        print(f"time: {time.time()-time_start:.0f}s", file=f)

        print(f"time_epoch: {time.time()-time_start_batch:.0f}s")
        print(f"time: {time.time()-time_start_batch:.0f}s", file=f)

        if (epoch + 1) % save_rate == 0:
            with torch.no_grad():
                dossier_midle = Path(
                    folder_result + f"/epoch{len(train_loss['total'])}"
                )
                dossier_midle.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    folder_result
                    + f"/epoch{len(train_loss['total'])}"
                    + "/model_weights.pth",
                )

                write_csv(
                    train_loss,
                    folder_result + f"/epoch{len(train_loss['total'])}",
                    file_name="/train_loss.csv",
                )

