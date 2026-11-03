# BBNet 재현 노트북 v8
"""BB-Net 학습·평가 (논문 표 6) 와 ablation 변형 (표 5, 그림 5). 세션이 끊겨도 이어서 학습한다."""
import json, math, os, time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import modeling_bbnet as mb
from modeling_bbnet import BBNet, featurise, collate, distance_loss, D_MIN, NUM_CLASSES
import bbrepro as R


# ------------------------------------------------------------------ 입력
def featurise_raw(code):
    """shift canonicalisation 없이 (표 5 'Shift-Canon ×')."""
    polys = code.supports
    ids, pos = mb.polynomial_tokens(polys, code.ell, code.m)
    return {"poly_ids": torch.from_numpy(ids), "poly_pos": torch.from_numpy(pos),
            "spectral": torch.from_numpy(mb.spectral_tokens(polys, code.ell, code.m)),
            "cond": torch.from_numpy(mb.conditioning(code))}


class CodeDataset(Dataset):
    def __init__(self, recs, augment, canon=True):
        self.codes = [R.code_from_record(r) for r in recs]
        self.dist = [int(r["d"]) for r in recs]
        self.augment, self.canon = augment, canon
        self.cache = None if augment else [self.feat(c) for c in self.codes]

    def feat(self, code):
        return featurise(code) if self.canon else featurise_raw(code)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        if self.augment:
            return self.feat(R.random_symmetry(self.codes[i], np.random.default_rng())), self.dist[i]
        return self.cache[i], self.dist[i]


def collate_labels(items):
    batch = collate([f for f, _ in items])
    batch["distance"] = torch.tensor([d for _, d in items], dtype=torch.long)
    return batch


# ------------------------------------------------------------------ 모델 변형
class PolyOnlyBBNet(BBNet):
    """Tanner 스펙트럼 가지가 없는 모델 (표 5 의 'BB-Net', 파라미터 수를 맞추려고 L = 16)."""
    def __init__(self, depth, width):
        super().__init__(depth, width)
        self.spectral = None
        self.gate = None

    def forward(self, batch):
        h = self.polynomial(batch["poly_ids"], batch["poly_pos"], batch["poly_mask"], batch["cond"])
        return self.head(h)


def make_model(tanner=True, film=True, depth=None, width=256):
    depth = depth or (8 if tanner else 16)
    model = BBNet(depth, width) if tanner else PolyOnlyBBNet(depth, width)
    if not film:
        for blk in model.polynomial.blocks:
            blk.film = None
    return model


def run_name(tanner, augment, canon, film, seed):
    return f"{'tanner' if tanner else 'poly'}_aug{int(augment)}_canon{int(canon)}_film{int(film)}_s{seed}"


# ------------------------------------------------------------------ 평가
def _amp(device):
    if device.type != "cuda":
        return False, None
    return True, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


@torch.no_grad()
def predict_proba(model, codes, device, canon=True, batch=1024):
    """코드 리스트 -> (확률 (N, NUM_CLASSES), 총 소요 초: featurise + forward)."""
    model.eval()
    use_amp, dt = _amp(device)
    t0 = time.time(); out = []
    for s in range(0, len(codes), batch):
        feats = [featurise(c) if canon else featurise_raw(c) for c in codes[s:s + batch]]
        b = {k: v.to(device, non_blocking=True) for k, v in collate(feats).items()}
        with torch.autocast(device.type, dtype=dt, enabled=use_amp):
            logits = model(b)
        out.append(torch.softmax(logits.float(), -1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(out).numpy(), time.time() - t0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    use_amp, dt = _amp(device)
    nll, P, G = 0.0, [], []
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast(device.type, dtype=dt, enabled=use_amp):
            logits = model(b)
        nll += distance_loss(logits.float(), b["distance"]).item() * len(logits)
        P.append(logits.argmax(-1).cpu() + D_MIN); G.append(b["distance"].cpu())
    p, g = torch.cat(P).float(), torch.cat(G).float()
    e = (p - g).abs()
    return dict(nll=nll / len(g), acc2=(e <= 2).float().mean().item(), exact=(e == 0).float().mean().item(),
                rmse=float(torch.sqrt(((p - g) ** 2).mean())))


# ------------------------------------------------------------------ 학습
def train_run(run_dir, train_recs, val_recs, *, tanner=True, film=True, augment=True, canon=True,
              depth=None, width=256, epochs=90, warmup_epochs=9, batch=256, lr=3e-4, wd=0.01,
              seed=0, num_workers=8, save_every=5, need_best=True,
              sched_type="cosine", clip=1.0, amp=True, wd_mode="all", log=print):
    """표 6 설정. run_dir 에 history.json / best.pt (검증 NLL 최소) / state.pt (재개용) 를 저장."""
    os.makedirs(run_dir, exist_ok=True)
    hist_path, state_path, best_path = (os.path.join(run_dir, f) for f in ("history.json", "state.pt", "best.pt"))
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    if len(history) >= epochs and (os.path.exists(best_path) or not need_best):   # ablation 은 기록만 있으면 충분
        log(f"[{os.path.basename(run_dir)}] 이미 {len(history)} 에폭 완료 — 건너뜀")
        return history
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(tanner, film, depth, width).to(device)
    kw = dict(batch_size=batch, collate_fn=collate_labels, num_workers=num_workers,
              pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    if num_workers > 0:
        kw["prefetch_factor"] = 4
    tl = DataLoader(CodeDataset(train_recs, augment, canon), shuffle=True, drop_last=len(train_recs) >= batch, **kw)
    vl = DataLoader(CodeDataset(val_recs, False, canon), shuffle=False, **kw)
    if wd_mode == "matrices":   # weight decay 를 행렬 파라미터에만 (bias·LayerNorm·게이트·CLS 제외)
        dec, nodec = [], []
        for n_, p_ in model.named_parameters():
            (nodec if p_.ndim < 2 or "cls" in n_ else dec).append(p_)
        opt = torch.optim.AdamW([{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}], lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    spe = len(tl); total, warm = epochs * spe, max(1, warmup_epochs * spe)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              (1.0 if sched_type == "constant" else
                                               0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
    use_amp, dt = _amp(device) if amp else (False, None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dt == torch.float16)
    start = 1
    if os.path.exists(state_path) and history:
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start = st["epoch"] + 1
        history = history[:st["epoch"]]
        log(f"[{os.path.basename(run_dir)}] {st['epoch']} 에폭부터 재개")
    best = min([h["val_nll"] for h in history], default=float("inf"))
    best_acc = max([h["val_acc2"] for h in history], default=-1.0)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{os.path.basename(run_dir)}] 파라미터 {n_params:,} | train {len(train_recs):,} | val {len(val_recs):,} | "
        f"{spe} step/에폭 | {device} {'(' + str(dt).split('.')[-1] + ')' if use_amp else ''}")
    for ep in range(start, epochs + 1):
        model.train(); t0 = time.time(); L = N = 0
        for b in tl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device.type, dtype=dt, enabled=use_amp):
                loss = distance_loss(model(b).float(), b["distance"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); sched.step()
            L += loss.item() * len(b["distance"]); N += len(b["distance"])
        v = evaluate(model, vl, device)
        gate = float(torch.tanh(model.gate.detach())) if getattr(model, "gate", None) is not None else float("nan")
        history.append(dict(epoch=ep, train_loss=L / max(N, 1), val_nll=v["nll"], val_acc2=v["acc2"],
                            val_exact=v["exact"], val_rmse=v["rmse"], lr=sched.get_last_lr()[0],
                            gate=gate, sec=time.time() - t0))
        flag = ""
        if v["nll"] < best:
            best = v["nll"]; flag = " *"
            torch.save({"state_dict": model.state_dict(), "tanner": tanner, "film": film, "canon": canon,
                        "depth": depth or (8 if tanner else 16), "width": width, "epoch": ep}, best_path)
        if v["acc2"] > best_acc:   # 검증 정확도(±2) 기준 체크포인트도 따로 저장
            best_acc = v["acc2"]
            torch.save({"state_dict": model.state_dict(), "tanner": tanner, "film": film, "canon": canon,
                        "depth": depth or (8 if tanner else 16), "width": width, "epoch": ep},
                       os.path.join(run_dir, "best_acc.pt"))
        if ep % save_every == 0 or ep == epochs:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep),
                       state_path)
            json.dump(history, open(hist_path, "w"))
        log(f"  ep {ep:3d} | train {L / max(N, 1):.3f} | val NLL {v['nll']:.3f} acc±2 {v['acc2']:.4f} "
            f"RMSE {v['rmse']:.3f} | tanh(g) {gate:+.3f} | {time.time() - t0:.0f}s{flag}")
    json.dump(history, open(hist_path, "w"))
    return history


def load_best(run_dir, device=None, name="best.pt"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(run_dir, name), map_location=device, weights_only=False)
    model = make_model(ck["tanner"], ck["film"], ck["depth"], ck["width"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck
