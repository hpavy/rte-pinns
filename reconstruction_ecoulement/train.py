from model import pde
import torch
import time
from utils import write_csv
from pathlib import Path
from torch.utils.data import DataLoader
from utils import CustomDataset

def train(
    nb_epoch,
    train_loss,
    test_loss,
    weight_data_init,
    weight_pde_init,
    weight_border_init,
    dynamic_weights,
    lr_weights,
    model,
    loss,
    optimizer,
    X_train,
    U_train,
    X_test_pde,
    X_test_data,
    U_test_data,
    X_pde,
    Re,
    time_start,
    f,
    folder_result,
    save_rate,
    batch_size,
    scheduler,
    X_border,
    X_border_test,
    mean_std,
    param_adim,
    nb_simu,
    force_inertie_bool,
):
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

    if device == torch.device("cuda"):
        stream_data = torch.cuda.Stream()
        stream_pde = torch.cuda.Stream()
        stream_border = torch.cuda.Stream()

    weight_border = torch.tensor(weight_border_init, dtype=torch.float32, device=device)
    weight_data = torch.tensor(weight_data_init, dtype=torch.float32, device=device)
    weight_pde = torch.tensor(weight_pde_init, dtype=torch.float32, device=device)

    nb_batches = len(X_pde) // batch_size
    # batch_size = torch.tensor(batch_size, device=device, dtype=torch.int64)

    Re = torch.tensor(Re, dtype=torch.float32, device=device)
    ya0_mean = mean_std["ya0_mean"].clone().to(device)
    ya0_std = mean_std["ya0_std"].clone().to(device)
    w0_mean = mean_std["w0_mean"].clone().to(device)
    w0_std = mean_std["w0_std"].clone().to(device)
    x_std = mean_std["x_std"].clone().to(device)
    y_std = mean_std["y_std"].clone().to(device)
    u_mean = mean_std["u_mean"].clone().to(device)
    v_mean = mean_std["v_mean"].clone().to(device)
    p_std = mean_std["p_std"].clone().to(device)
    t_std = mean_std["t_std"].clone().to(device)
    t_mean = mean_std["t_mean"].clone().to(device)
    u_std = mean_std["u_std"].clone().to(device)
    v_std = mean_std["v_std"].clone().to(device)
    L_adim = torch.tensor(param_adim["L"], device=device, dtype=torch.float32)
    V_adim = torch.tensor(param_adim["V"], device=device, dtype=torch.float32)
    X_border = X_border.to(device)
    X_border_test = X_border_test.to(device).detach()
    ########
    X_pde = X_pde.to(device)
    ########
    # dataset = CustomDataset(X_pde)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    ###############
    X_test_pde = X_test_pde.to(device).requires_grad_(True)
    X_train = X_train.to(device)
    U_train = U_train.to(device)
    X_test_data = X_test_data.to(device).detach()
    U_test_data = U_test_data.to(device).detach()
    nb_simu = torch.tensor(nb_simu, device=device, dtype=torch.int64)
    len_X_train_one = (
        torch.tensor(X_train.size(0), device=device, dtype=torch.int64) // nb_simu
    )
    for epoch in range(len(train_loss["total"]), nb_it_tot):
        time_start_batch = time.time()
        total_batch = torch.tensor([0.0], device=device)
        data_batch = torch.tensor([0.0], device=device)
        pde_batch = torch.tensor([0.0], device=device)
        border_batch = torch.tensor([0.0], device=device)
        model.train()  # on dit qu'on va entrainer (on a le dropout)
        for nb_batch in range(nb_batches):
        # for nb_batch, X_pde_batch in enumerate(dataloader):
            with torch.cuda.stream(stream_pde):
                # X_pde_batch = X_pde_batch.to(device).requires_grad_(True)
                # loss du pde
                X_pde_batch = (
                    X_pde[nb_batch * batch_size: (nb_batch + 1) * batch_size, :]
                    .clone()
                    .requires_grad_(True)
                ).to(device)
                pred_pde = model(X_pde_batch)
                pred_pde1, pred_pde2, pred_pde3, res, y_ = pde(
                    pred_pde,
                    X_pde_batch,
                    Re=Re,
                    x_std=x_std,
                    y_std=y_std,
                    u_mean=u_mean,
                    v_mean=v_mean,
                    p_std=p_std,
                    t_std=t_std,
                    t_mean=t_mean,
                    u_std=u_std,
                    v_std=v_std,
                    ya0_mean=ya0_mean,
                    ya0_std=ya0_std,
                    w0_mean=w0_mean,
                    w0_std=w0_std,
                    L_adim=L_adim,
                    V_adim=V_adim,
                    force_inertie_bool=force_inertie_bool,
                )
                loss_pde = (
                    torch.mean(pred_pde1**2)
                    + torch.mean(pred_pde2**2) * 0.
                    + torch.mean(pred_pde3**2)
                )

            with torch.cuda.stream(stream_data):
                X_train_batch = (
                    X_train[
                        (nb_batch % nb_simu)
                        * len_X_train_one : (nb_batch % nb_simu + 1)
                        * len_X_train_one
                    ]
                    .clone()
                    .requires_grad_()
                )
                U_train_batch = (
                    U_train[
                        (nb_batch % nb_simu)
                        * len_X_train_one : (nb_batch % nb_simu + 1)
                        * len_X_train_one
                    ]
                    .clone()
                    .requires_grad_()
                )
                # loss des points de data
                pred_data = model(X_train_batch)
                loss_data = loss(U_train_batch, pred_data)

            with torch.cuda.stream(stream_border):
                # loss du border
                pred_border = model(X_border)
                goal_border = torch.tensor(
                    [
                        -mean_std["u_mean"] / mean_std["u_std"],
                        -mean_std["v_mean"] / mean_std["v_std"],
                    ],
                    dtype=torch.float32,
                    device=device,
                ).expand(pred_border.shape[0], 2)
                loss_border_cylinder = loss(pred_border[:, :2], goal_border)  # (MSE)

            torch.cuda.synchronize()

            loss_totale = (
                weight_data * loss_data
                + weight_pde * loss_pde
                + weight_border * loss_border_cylinder
            )

            # Backpropagation
            loss_totale.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                total_batch += loss_totale.item()
                data_batch += loss_data.item()
                pde_batch += loss_pde.item()
                border_batch += loss_border_cylinder.item()

        ######## A partir d'ici #######
        # Pour le test :
        model.eval()

        # loss du pde
        test_pde = model(X_test_pde)
        test_pde1, test_pde2, test_pde3, res, y_ = pde(
            test_pde,
            X_test_pde,
            Re=Re,
            x_std=x_std,
            y_std=y_std,
            u_mean=u_mean,
            v_mean=v_mean,
            p_std=p_std,
            t_std=t_std,
            t_mean=t_mean,
            u_std=u_std,
            v_std=v_std,
            ya0_mean=ya0_mean,
            ya0_std=ya0_std,
            w0_mean=w0_mean,
            w0_std=w0_std,
            L_adim=L_adim,
            V_adim=V_adim,
            force_inertie_bool=force_inertie_bool,
        )
        with torch.no_grad():
            loss_test_pde = (
                torch.mean(test_pde1**2)
                + torch.mean(test_pde2**2) * 0.
                + torch.mean(test_pde3**2)
            )
            # loss de la data
            test_data = model(X_test_data)
            loss_test_data = loss(U_test_data, test_data)  # (MSE)

            # loss des bords
            pred_border_test = model(X_border_test)
            goal_border_test = torch.tensor(
                [
                    -mean_std["u_mean"] / mean_std["u_std"],
                    -mean_std["v_mean"] / mean_std["v_std"],
                ],
                dtype=torch.float32,
                device=device,
            ).expand(pred_border_test.shape[0], 2)
            loss_test_border = loss(pred_border_test[:, :2], goal_border_test)  # (MSE)

            # loss totale
            loss_test = (
                weight_data * loss_test_data
                + weight_pde * loss_test_pde
                + weight_border * loss_test_border
            )
        scheduler.step()

        # Weights
        with torch.no_grad():
            total_batch /= nb_batch
            data_batch /= nb_batch
            pde_batch /= nb_batch
            border_batch /= nb_batch
            if dynamic_weights:
                weight_data_hat = weight_data + lr_weights * data_batch
                weight_pde_hat = weight_pde + lr_weights * pde_batch
                weight_border_hat = weight_border + lr_weights * border_batch
                sum_weight = weight_data_hat + weight_pde_hat + weight_border_hat 
                weight_data = weight_data_hat / sum_weight
                weight_border = weight_border_hat / sum_weight
                weight_pde = weight_pde_hat / sum_weight
            test_loss["total"].append(loss_test.item())
            test_loss["data"].append(loss_test_data.item())
            test_loss["pde"].append(loss_test_pde.item())
            test_loss["border"].append(loss_test_border.item())
            train_loss["total"].append(total_batch.item())
            train_loss["data"].append(data_batch.item())
            train_loss["pde"].append(pde_batch.item())
            train_loss["border"].append(border_batch.item())
            # train_loss["border"].append(-1)

        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}"
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}",
            file=f,
        )
        print(
            f"Weights  : ------------, data: {weight_data.item():.1e},   pde: {weight_pde.item():.1e},   border: {weight_border.item():.1e}"
        )
        print(
            f"Weights  : ------------, data: {weight_data.item():.1e},   pde: {weight_pde.item():.1e},   border: {weight_border.item():.1e}",
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
                        "weights": {
                            "weight_data": weight_data,
                            "weight_pde": weight_pde,
                            "weight_border": weight_border,
                        },
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
                write_csv(
                    test_loss,
                    folder_result + f"/epoch{len(train_loss['total'])}",
                    file_name="/test_loss.csv",
                )
