import argparse
import gc
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from common._mlflow import get_callbacks, get_mlflow_logger
from common.datasets import TripletDatasetFromCached, UserItemPairDatasetFromCached
from common.utils import seed_worker, set_seed
from lightning_models.extensions.mtdp_kgat_v2 import MTDPRec
from pytorch_lightning import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="MTDP KGAT Training")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--rel_emb_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_weight", type=float, default=1e-5)
    parser.add_argument("--experiment_name", type=str, default="mtdp-kgat-v2-loss-weight-exp")
    parser.add_argument("--version", type=str, default="baseline")
    parser.add_argument("--run_name", type=str, default="run00")
    parser.add_argument("--rec_loss", type=float, default=1.0)
    parser.add_argument("--dps_loss", type=float, default=1.0)
    parser.add_argument("--dpr_loss", type=float, default=1.0)
    parser.add_argument("--dpm_loss", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    RANDOM_SEED = 42
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience
    USE_MINI_BATCH = False
    NODE_DROPOUT_RATE = 0
    MESS_DROPOUT_RATE = 0.2

    set_seed(RANDOM_SEED)
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    cache = args.cache_dir

    train_dataset = TripletDatasetFromCached(split="train")
    print("train data count:", len(train_dataset))
    valid_dataset = TripletDatasetFromCached(split="valid")
    print("valid data count:", len(valid_dataset))
    test_dataset = UserItemPairDatasetFromCached(split="prediction")
    print("test data count:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    train_hetero_graph = torch.load(f"{cache}/train_hetero_graph.pt", weights_only=False)

    gc.collect()

    DPS_WEIGHTS = {k: 0.25 for k in ["actor_dps", "country_dps", "director_dps", "genre_dps"]}
    DPR_WEIGHTS = {k: 0.25 for k in ["actor_dpr", "country_dpr", "director_dpr", "genre_dpr"]}
    DPM_WEIGHTS = {k: 0.25 for k in ["actor_pd", "country_pd", "director_pd", "genre_pd"]}
    MT_WEIGHTS = {
        "rec_loss": args.rec_loss,
        "dps_loss": args.dps_loss,
        "dpr_loss": args.dpr_loss,
        "dpm_loss": args.dpm_loss,
    }

    model = MTDPRec(
        hetero_data=train_hetero_graph,
        embedding_dim=args.embedding_dim,
        rel_emb_dim=args.rel_emb_dim,
        num_layers=args.num_layers,
        node_dropout=NODE_DROPOUT_RATE,
        mess_dropout=MESS_DROPOUT_RATE,
        num_neighbors=10,
        lr=args.lr,
        reg_weight=args.reg_weight,
        use_mini_batch=USE_MINI_BATCH,
        dps_weights=DPS_WEIGHTS,
        dpr_weights=DPR_WEIGHTS,
        dpm_weights=DPM_WEIGHTS,
        mt_weights=MT_WEIGHTS,
    )

    mlflow_logger = get_mlflow_logger(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tags={"version": args.version},
    )

    trainer_callbacks = get_callbacks(
        exp_name=args.experiment_name,
        version_name=args.version,
        run_name=args.run_name,
        patience=PATIENCE,
        monitor_metric="val_loss",
        monitor_mode="min",
        hyper_param_str=f"emb_dim={args.embedding_dim}-num_layers={args.num_layers}-lr={args.lr}",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=mlflow_logger,
        log_every_n_steps=50,
        callbacks=trainer_callbacks,
        accelerator="gpu",
        devices=[int(args.device)],
    )

    gc.collect()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    gc.collect()

    best_model_path = trainer.checkpoint_callback.best_model_path
    model = MTDPRec.load_from_checkpoint(checkpoint_path=best_model_path)
    trainer.test(model=model, dataloaders=test_loader)
    print(model.test_results["eval_score_df"].describe())


if __name__ == "__main__":
    main()
    main()
