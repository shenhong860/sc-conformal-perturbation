"""Fine-tune scGPT (whole-human) for Norman perturbation prediction."""

import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from gears import PertData
from torch import nn

import scgpt as scg
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id, set_seed

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
warnings.filterwarnings("ignore")
set_seed(SEED)

DATA_DIR = Path("/mnt/d/guo/CW1OT/results/scgpt_norman/data_union")
SPLITS = Path("/mnt/d/guo/CW1OT/results/scgpt_norman/splits_union.json")
WEIGHTS = Path("/mnt/d/guo/CW1OT/externals/scgpt_weights")
SAVE_DIR = Path(f"/mnt/d/guo/CW1OT/results/scgpt_norman/finetune_seed{SEED}")

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

MLM = True
CLS = False
CCE = False
MVC = False
ECS = False
amp = True
load_param_prefixs = ["encoder", "value_encoder", "transformer_encoder"]

lr = 1e-4
batch_size = 64
eval_batch_size = 64
epochs = 4
schedule_interval = 1
early_stop = 3
log_interval = 50
dropout = 0
use_fast_transformer = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pertdata():
    pert_data = PertData(str(DATA_DIR))
    pert_data.load(data_path=str(DATA_DIR / "norman_union"))
    splits = json.loads(SPLITS.read_text())
    pert_data.split = "custom"
    pert_data.set2conditions = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
    return pert_data, splits["genes"]


def build_model(pert_data):
    vocab = GeneVocab.from_file(WEIGHTS / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes],
        dtype=int,
    )
    configs = json.loads((WEIGHTS / "args.json").read_text())
    model = TransformerGenerator(
        len(vocab),
        configs["embsize"],
        configs["nheads"],
        configs["d_hid"],
        configs["nlayers"],
        nlayers_cls=configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )
    model_dict = model.state_dict()
    pretrained = torch.load(WEIGHTS / "best_model.pt", map_location="cpu", weights_only=False)
    pretrained = {
        k: v
        for k, v in pretrained.items()
        if any(k.startswith(p) for p in load_param_prefixs)
        and k in model_dict
        and v.shape == model_dict[k].shape
    }
    print("loading pretrained params:", len(pretrained))
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model.to(device)
    return model, vocab, gene_ids


def train_one_epoch(model, loader, gene_ids, criterion, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    n_genes = len(gene_ids)
    start = time.time()
    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        batch_data.to(device)
        x = batch_data.x
        ori_gene_values = x[:, 0].view(batch_data.num_graphs, n_genes)
        pert_flags = x[:, 1].long().view(batch_data.num_graphs, n_genes)
        target_gene_values = batch_data.y

        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        if len(input_gene_ids) > max_seq_len:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]
        mapped = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(batch_data.num_graphs, 1)
        src_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=amp):
            output = model(
                mapped,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            loss = criterion(
                output["mlm_output"],
                target_values,
                torch.ones_like(input_values, dtype=torch.bool),
            )

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            ms = (time.time() - start) * 1000 / log_interval
            print(
                f"| epoch {epoch} | {batch}/{num_batches} | lr {scheduler.get_last_lr()[0]:.5f} "
                f"| ms/batch {ms:5.1f} | loss {total_loss / log_interval:.4f} |",
                flush=True,
            )
            total_loss = 0.0
            start = time.time()


def eval_perturb(loader, model, gene_ids):
    model.eval()
    pert_cat, pred, truth = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            pert_cat.extend(batch.pert)
            p = model.pred_perturb(
                batch, include_zero_gene=include_zero_gene, gene_ids=gene_ids
            )
            pred.append(p.cpu())
            truth.append(batch.y.cpu())
    return {
        "pert_cat": np.array(pert_cat),
        "pred": torch.cat(pred).numpy(),
        "truth": torch.cat(truth).numpy(),
    }


def pearson_by_condition(res, ctrl_mean=None, use_logfc=False):
    conds = {}
    for p, pr, tr in zip(res["pert_cat"], res["pred"], res["truth"]):
        conds.setdefault(p, [[], []])
        conds[p][0].append(pr)
        conds[p][1].append(tr)
    corrs = []
    for p, (prs, trs) in conds.items():
        pm = np.mean(prs, axis=0)
        tm = np.mean(trs, axis=0)
        if use_logfc:
            pm = pm - ctrl_mean
            tm = tm - ctrl_mean
        corrs.append(float(np.corrcoef(pm, tm)[0, 1]))
    return float(np.mean(corrs))


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger = scg.logger
    scg.utils.add_file_handler(logger, SAVE_DIR / "run.log")
    pert_data, genes = load_pertdata()
    model, vocab, gene_ids = build_model(pert_data)
    n_genes = len(genes)

    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_corr = -1.0
    best_model = None
    patience = 0
    ctrl_mean = np.asarray(
        pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].X.mean(axis=0)
    ).ravel()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_one_epoch(
            model,
            pert_data.dataloader["train_loader"],
            gene_ids,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
        )
        val_res = eval_perturb(pert_data.dataloader["val_loader"], model, gene_ids)
        corr = pearson_by_condition(val_res)
        corr_lfc = pearson_by_condition(val_res, ctrl_mean, use_logfc=True)
        print(f"epoch {epoch} val pearson {corr:.4f} time {time.time()-t0:.1f}s", flush=True)
        print(f"epoch {epoch} val logFC pearson {corr_lfc:.4f}", flush=True)
        if corr > best_corr:
            best_corr = corr
            best_model = copy.deepcopy(model)
            patience = 0
            print(f"best model {corr:.4f}", flush=True)
        else:
            patience += 1
            if patience >= early_stop:
                break
        scheduler.step()

    torch.save(best_model.state_dict(), SAVE_DIR / "best_model.pt")
    test_res = eval_perturb(pert_data.dataloader["test_loader"], best_model, gene_ids)
    np.savez(
        SAVE_DIR / "test_res.npz",
        pert_cat=test_res["pert_cat"],
        pred=test_res["pred"],
        truth=test_res["truth"],
    )
    ctrl = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    ctrl_mean = np.asarray(ctrl.X.mean(axis=0)).ravel()
    np.save(SAVE_DIR / "ctrl_mean.npy", ctrl_mean)
    (SAVE_DIR / "genes.json").write_text(json.dumps(genes), encoding="utf-8")
    (SAVE_DIR / "val_pearson.json").write_text(json.dumps({"best": best_corr}), encoding="utf-8")
    print("saved best model and test results", flush=True)


if __name__ == "__main__":
    main()
