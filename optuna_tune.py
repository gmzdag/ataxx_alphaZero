import optuna
import mlflow
import time
from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper
from Coach import Coach
from utils import dotdict

def objective(trial):
    # 🔹 Hiperparametre aralıkları
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    num_channels = trial.suggest_categorical("num_channels", [64, 128, 256, 512])
    cpuct = trial.suggest_float("cpuct", 0.5, 2.0)
    epochs = trial.suggest_int("epochs", 5, 20)

    args = dotdict({
        'lr': lr,
        'dropout': dropout,
        'epochs': epochs,             
        'batch_size': 64,
        'num_channels': num_channels,
        'cuda': True,
        'numMCTSSims': 25,
        'cpuct': cpuct,
        'numIters': 1,
        'numEps': 5,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 10000,
        'arenaCompare': 20,
        'checkpoint': './temp/',
        'numItersForTrainExamplesHistory': 1
    })

    mlflow.set_experiment("Ataxx_Optuna_Tuning")

    with mlflow.start_run(run_name=f"trial_{trial.number}"):

        # 🔸 parametreleri logla
        mlflow.log_params({
            "lr": lr,
            "dropout": dropout,
            "num_channels": num_channels,
            "cpuct": cpuct,
            "epochs": epochs
        })

        # 🔸 model & oyun
        game = AtaxxGame(7)
        nnet = NNetWrapper(game)
        coach = Coach(game, nnet, args)

        # eğitim süresini ölç
        start_time = time.time()
        coach.learn()
        train_time = time.time() - start_time

        # 🔸 metrikleri topla (Coach içinden çek)
        win_rate = getattr(coach, "last_winrate", 0.5)
        avg_policy_loss = getattr(coach, "avg_policy_loss", None)
        avg_value_loss = getattr(coach, "avg_value_loss", None)
        training_examples = getattr(coach, "train_example_count", None)
        total_games_played = getattr(coach, "games_played", None)

        # 🔹 tüm metrikleri logla
        mlflow.log_metric("win_rate", win_rate)
        if avg_policy_loss is not None:
            mlflow.log_metric("avg_policy_loss", avg_policy_loss)
        if avg_value_loss is not None:
            mlflow.log_metric("avg_value_loss", avg_value_loss)
        if training_examples is not None:
            mlflow.log_metric("train_example_count", training_examples)
        if total_games_played is not None:
            mlflow.log_metric("games_played", total_games_played)

        mlflow.log_metric("train_time_sec", train_time)

        return win_rate


# 🔹 Optuna çalıştır
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("\n✨ En iyi parametreler bulundu:")
for k, v in study.best_params.items():
    print(f"{k}: {v:.4f}")
