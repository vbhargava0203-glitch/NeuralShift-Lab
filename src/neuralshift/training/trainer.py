"""Train once per dataset/model/seed; select and calibrate with validation only."""

import logging
import random
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuralshift.datasets.loaders import make_loader, metadata, training_data
from neuralshift.models.factory import build_model
from neuralshift.reproducibility import environment, seed_everything, select_device, timestamp
from neuralshift.training.checkpoints import load_checkpoint, save_checkpoint
from neuralshift.training.early_stopping import EarlyStopping
from neuralshift.training.losses import classification_loss
from neuralshift.uncertainty.temperature_scaling import fit_temperature
from neuralshift.utils.artifacts import atomic_csv, atomic_json, atomic_npz, read_json, sha256


def make_model(config):
    return build_model(config["model"], in_channels=1 if config["dataset"] == "fashion_mnist" else 3,
                       num_classes=10, **config["model_options"])


def epoch_pass(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = classification_loss()
    total_loss, correct, n, gradients = 0.0, 0, 0, []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; training aborted")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite gradients; training aborted")
                gradients.append(float(norm))
                optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        n += len(labels)
    return total_loss / n, correct / n, float(np.mean(gradients)) if gradients else 0.0


def collect_logits(model, loader, device):
    model.eval()
    logits, labels, ids = [], [], []
    with torch.inference_mode():
        for x, y, index in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
            ids.append(index)
    return torch.cat(logits), torch.cat(labels), torch.cat(ids)


def train(config, resume=None):
    seed_everything(config["seed"], config["training"]["deterministic"], config["training"]["threads"])
    device = select_device(config["training"]["device"])
    run = Path(resume) if resume else Path(config["output"]) / "runs" / uuid.uuid4().hex[:12]
    if resume:
        manifest = read_json(run / "manifest.json")
        if manifest["config"] != config:
            raise ValueError("Resume requires the identical resolved configuration")
        if manifest["status"] in {"training_complete", "evaluating", "evaluation_failed", "evaluation_complete", "validated"}:
            return run
    else:
        run.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "run_id": run.name, "config": config,
                    "created_at": timestamp(), "status": "training", "environment": environment(),
                    "device": str(device), "research_status": "smoke" if config["profile"] == "smoke" else "pilot" if config["profile"] == "development" else "research",
                    "protocol": "fixed deterministic predictor; all error scores target its mistakes; no test-label fitting"}
        atomic_json(run / "config.json", config)
        atomic_json(run / "manifest.json", manifest)
    try:
        train_set, val_set = training_data(config)
        manifest["dataset"] = metadata(config, train_set, val_set)
        atomic_json(run / "dataset.json", manifest["dataset"])
        generator = torch.Generator().manual_seed(config["seed"])
        train_loader = make_loader(train_set, config, shuffle=True, generator=generator)
        val_loader = make_loader(val_set, config, generator=torch.Generator().manual_seed(config["seed"] + 1))
        model = make_model(config).to(device)
        manifest["parameter_count"] = sum(p.numel() for p in model.parameters())
        atomic_json(run / "manifest.json", manifest)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
        stopping = EarlyStopping(config["training"]["patience"])
        history, start = [], 0
        if resume and (run / "last.pt").exists():
            state = load_checkpoint(run / "last.pt", device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            stopping.best, stopping.bad_epochs = state["best"], state["bad_epochs"]
            start, history = state["epoch"], state["history"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            generator.set_state(state["loader_rng"].cpu())
            if torch.cuda.is_available() and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        for epoch in range(start, config["training"]["epochs"]):
            if stopping.bad_epochs >= stopping.patience:
                break
            started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy, grad_norm = epoch_pass(model, train_loader, device, optimizer)
            val_loss, val_accuracy, _ = epoch_pass(model, val_loader, device)
            improved, stop = stopping.update(val_loss)
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                   "train_accuracy": train_accuracy, "val_accuracy": val_accuracy, "lr": lr,
                   "epoch_seconds": time.perf_counter() - started, "grad_norm": grad_norm}
            history.append(row)
            if improved:
                save_checkpoint(run / "best.pt", {"model": model.state_dict(), "config": config, "epoch": epoch + 1,
                                                   "validation_loss": val_loss, "selection_split": "validation"})
                manifest["best_epoch"] = epoch + 1
            save_checkpoint(run / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "history": history,
                            "best": stopping.best, "bad_epochs": stopping.bad_epochs,
                            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                            "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
            atomic_csv(run / "history.csv", pd.DataFrame(history))
            atomic_json(run / "manifest.json", manifest)
            logging.info("%s epoch %d train loss %.4f val loss %.4f val accuracy %.4f", run.name, epoch + 1, train_loss, val_loss, val_accuracy)
            if stop:
                break
        best = load_checkpoint(run / "best.pt", device)
        model.load_state_dict(best["model"])
        manifest["best_epoch"] = best["epoch"]
        logits, labels, ids = collect_logits(model, val_loader, device)
        temperature = float(fit_temperature(logits, labels, split="validation"))
        atomic_npz(run / "validation_logits.npz", logits=logits.numpy(), labels=labels.numpy(), ids=ids.numpy())
        atomic_json(run / "temperature.json", {"temperature": temperature, "split": "validation", "n": len(labels),
                    "validation_logits_sha256": sha256(run / "validation_logits.npz"), "checkpoint_sha256": sha256(run / "best.pt")})
        manifest.update(status="training_complete", training_finished_at=timestamp(),
                        checkpoint={"path": "best.pt", "sha256": sha256(run / "best.pt")})
        atomic_json(run / "manifest.json", manifest)
        return run
    except Exception as exc:
        manifest.update(status="training_failed", error=str(exc))
        atomic_json(run / "manifest.json", manifest)
        raise
