
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runnable synthetic benchmark for:
"A Dynamic Two-Tower Recommendation Model Fusing Digital Profiles and Semantic Features"

Features:
- Controlled synthetic educational benchmark
- Fast default configuration (600 users / 400 items); use command-line args for the paper-scale 1200 / 800 setting
- 5 methods comparison:
    1) Random
    2) Popularity
    3) ContentMatch
    4) StaticTT
    5) DynamicTT
- 5 alpha settings comparison for DynamicTT
- Outputs:
    * method_comparison.csv
    * alpha_sensitivity.csv
    * fig_method_comparison.png
    * fig_alpha_sensitivity.png
    * summary.txt

Dependencies:
    numpy, pandas, matplotlib
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_rows(x, eps: float = 1e-9):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(nrm, eps)


# -----------------------------
# Synthetic benchmark
# -----------------------------
@dataclass
class SyntheticData:
    P: np.ndarray           # (n_users, d_p)
    H: np.ndarray           # (n_users, d_h)
    Q: np.ndarray           # (n_items, d_q)
    levels: np.ndarray      # (n_users,)
    difficulty: np.ndarray  # (n_items,)
    seen_sets: list         # list[set[int]]
    target_sets: list       # list[set[int]]
    train_users: np.ndarray
    test_users: np.ndarray
    item_topics: np.ndarray # (n_items, n_topics)
    user_topic_pref: np.ndarray  # (n_users, n_topics)


def generate_synthetic_benchmark(
    n_users: int = 600,
    n_items: int = 400,
    n_windows: int = 5,
    interactions_per_window: int = 12,
    d_p: int = 13,
    d_q: int = 15,
    n_topics: int = 5,
    latent_true_dim: int = 16,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> SyntheticData:
    rng = set_seed(seed)

    # ----- item side -----
    # topics
    item_topic_idx = rng.integers(0, n_topics, size=n_items)
    item_topics = np.zeros((n_items, n_topics), dtype=np.float64)
    item_topics[np.arange(n_items), item_topic_idx] = 1.0

    # difficulty 1..6
    difficulty = rng.integers(1, 7, size=n_items)

    # latent semantics and observable semantic features
    Z = rng.normal(0, 1, size=(n_items, latent_true_dim))
    A_q = rng.normal(0, 1, size=(latent_true_dim, 10))
    semantic_dense = Z @ A_q + 0.15 * rng.normal(size=(n_items, 10))
    semantic_dense = (semantic_dense - semantic_dense.mean(0)) / (semantic_dense.std(0) + 1e-9)

    # q_i: 5 topic dims + 10 semantic dims = 15
    Q = np.concatenate([item_topics, semantic_dense], axis=1).astype(np.float64)

    # ----- user side -----
    grades = rng.integers(1, 5, size=n_users)                # 1..4
    grade_onehot = np.zeros((n_users, 4), dtype=np.float64)
    grade_onehot[np.arange(n_users), grades - 1] = 1.0

    # user topic preference
    user_topic_pref = rng.dirichlet(alpha=np.ones(n_topics) * 1.6, size=n_users)

    # reading level roughly tracks grade, but with noise
    levels = np.clip(np.round(grades + rng.normal(1.2, 0.8, size=n_users)), 1, 6).astype(int)

    # extra profile features so d_p = 4 + 5 + 1 + 3 = 13
    X_misc = rng.normal(0, 1, size=(n_users, 3))
    X_misc = (X_misc - X_misc.mean(0)) / (X_misc.std(0) + 1e-9)
    P = np.concatenate(
        [grade_onehot, user_topic_pref, (levels / 6.0)[:, None], X_misc],
        axis=1,
    ).astype(np.float64)

    # latent user preference basis
    U_base = rng.normal(0, 1, size=(n_users, latent_true_dim))
    U_base = normalize_rows(U_base)

    # random drift directions
    drift_dirs = rng.normal(0, 1, size=(n_users, latent_true_dim))
    drift_dirs = normalize_rows(drift_dirs)

    # temporal topic shift preferences
    topic_shift = rng.normal(0, 0.35, size=(n_users, n_topics))
    user_window_history = [[] for _ in range(n_users)]
    seen_sets = [set() for _ in range(n_users)]

    for t in range(n_windows):
        drift_strength = 0.18 * t
        U_t = normalize_rows(U_base + drift_strength * drift_dirs)

        topic_pref_t = user_topic_pref + (t / max(n_windows - 1, 1)) * topic_shift
        topic_pref_t = np.clip(topic_pref_t, 0.01, None)
        topic_pref_t = topic_pref_t / topic_pref_t.sum(axis=1, keepdims=True)

        for u in range(n_users):
            unread_mask = np.ones(n_items, dtype=bool)
            if seen_sets[u]:
                unread_mask[list(seen_sets[u])] = False

            # latent affinity + topic consistency + difficulty suitability + noise
            latent_score = Z @ U_t[u]
            topic_score = item_topics @ topic_pref_t[u]
            diff_penalty = -0.55 * np.abs(difficulty - levels[u])
            grade_bonus = 0.15 * grades[u] * (item_topic_idx == np.argmax(topic_pref_t[u]))
            noise = rng.normal(0, 0.18, size=n_items)

            score = 1.05 * latent_score + 0.90 * topic_score + diff_penalty + grade_bonus + noise
            score[~unread_mask] = -1e18

            chosen = np.argpartition(score, -interactions_per_window)[-interactions_per_window:]
            chosen = chosen[np.argsort(score[chosen])[::-1]]
            chosen = chosen.tolist()

            user_window_history[u].append(chosen)
            seen_sets[u].update(chosen)

    # target is window 5; history summary comes from windows 1..4 with recency weights
    history_weights = np.array([0.12, 0.18, 0.28, 0.42], dtype=np.float64)
    H = np.zeros((n_users, d_q), dtype=np.float64)
    target_sets = []
    seen_before_target = []

    for u in range(n_users):
        weighted = np.zeros(d_q, dtype=np.float64)
        total_w = 0.0
        seen_hist = set()
        for t in range(n_windows - 1):
            ids = user_window_history[u][t]
            if ids:
                weighted += history_weights[t] * Q[ids].mean(axis=0)
                total_w += history_weights[t]
                seen_hist.update(ids)
        if total_w > 0:
            H[u] = weighted / total_w
        target_sets.append(set(user_window_history[u][-1]))
        seen_before_target.append(seen_hist)

    perm = rng.permutation(n_users)
    cut = int(train_ratio * n_users)
    train_users = np.sort(perm[:cut])
    test_users = np.sort(perm[cut:])

    return SyntheticData(
        P=P,
        H=H,
        Q=Q,
        levels=levels,
        difficulty=difficulty,
        seen_sets=seen_before_target,
        target_sets=target_sets,
        train_users=train_users,
        test_users=test_users,
        item_topics=item_topics,
        user_topic_pref=user_topic_pref,
    )


# -----------------------------
# Metrics
# -----------------------------
def dcg_at_k(relevance):
    relevance = np.asarray(relevance, dtype=np.float64)
    if relevance.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance * discounts))


def evaluate_rankings(rankings, target_sets, levels, difficulty, tau, ks=(5, 10, 20)):
    rows = []
    users = sorted(rankings.keys())
    for K in ks:
        hit_list, recall_list, ndcg_list, dmr_list = [], [], [], []
        for u in users:
            preds = rankings[u][:K]
            truth = target_sets[u]
            if len(truth) == 0:
                continue
            rel = np.array([1 if i in truth else 0 for i in preds], dtype=np.float64)
            hit = 1.0 if rel.sum() > 0 else 0.0
            recall = rel.sum() / len(truth)
            idcg = dcg_at_k(np.ones(min(K, len(truth)), dtype=np.float64))
            ndcg = dcg_at_k(rel) / idcg if idcg > 0 else 0.0
            dmr = np.mean([1.0 if abs(int(difficulty[i]) - int(levels[u])) <= tau else 0.0 for i in preds]) if preds else 0.0

            hit_list.append(hit)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            dmr_list.append(dmr)

        rows.append({
            "K": K,
            "HitRate": float(np.mean(hit_list)),
            "Recall": float(np.mean(recall_list)),
            "NDCG": float(np.mean(ndcg_list)),
            "DMR": float(np.mean(dmr_list)),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Baselines
# -----------------------------
def recommend_random(data: SyntheticData, candidate_users, n_recommend: int = 20, seed: int = 123):
    rng = np.random.default_rng(seed)
    rankings = {}
    all_items = np.arange(data.Q.shape[0])
    for u in candidate_users:
        mask = np.ones_like(all_items, dtype=bool)
        if data.seen_sets[u]:
            mask[list(data.seen_sets[u])] = False
        candidates = all_items[mask]
        chosen = rng.choice(candidates, size=min(n_recommend, len(candidates)), replace=False)
        rankings[u] = chosen.tolist()
    return rankings


def recommend_popularity(data: SyntheticData, candidate_users, n_recommend: int = 20):
    counts = np.zeros(data.Q.shape[0], dtype=np.int64)
    for u in data.train_users:
        for i in data.seen_sets[u]:
            counts[i] += 1
    global_order = np.argsort(counts)[::-1]
    rankings = {}
    for u in candidate_users:
        preds = [int(i) for i in global_order if i not in data.seen_sets[u]][:n_recommend]
        rankings[u] = preds
    return rankings


def recommend_contentmatch(data: SyntheticData, candidate_users, n_recommend: int = 20, beta: float = 0.65):
    rankings = {}
    topic_part = data.Q[:, :data.item_topics.shape[1]]
    sem_part = data.Q[:, data.item_topics.shape[1]:]
    H_sem = data.H[:, data.item_topics.shape[1]:]

    for u in candidate_users:
        # explicit topic preference + recent semantic similarity - difficulty mismatch
        topic_score = topic_part @ data.user_topic_pref[u]
        history_score = sem_part @ H_sem[u]
        history_score = history_score / (np.linalg.norm(H_sem[u]) + 1e-9)
        diff_penalty = beta * np.abs(data.difficulty - data.levels[u])
        score = 0.65 * topic_score + 0.55 * history_score - diff_penalty

        if data.seen_sets[u]:
            score[list(data.seen_sets[u])] = -1e18
        order = np.argsort(score)[::-1][:n_recommend]
        rankings[u] = order.tolist()
    return rankings


# -----------------------------
# Two-tower model
# -----------------------------
class LinearTwoTower:
    def __init__(
        self,
        d_p: int,
        d_h: int,
        d_q: int,
        latent_dim: int = 32,
        lr: float = 0.02,
        reg: float = 1e-4,
        alpha: float = 1.5,
        use_history: bool = True,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        scale = 0.12
        self.Wp = rng.normal(0, scale, size=(latent_dim, d_p))
        self.Wh = rng.normal(0, scale, size=(latent_dim, d_h))
        self.Wq = rng.normal(0, scale, size=(latent_dim, d_q))
        self.bu = np.zeros(latent_dim, dtype=np.float64)
        self.bi = np.zeros(latent_dim, dtype=np.float64)
        self.lr = lr
        self.reg = reg
        self.alpha = alpha
        self.use_history = use_history

    def user_emb(self, p, h):
        out = self.Wp @ p + self.bu
        if self.use_history:
            out = out + self.Wh @ h
        return out

    def item_emb(self, q):
        return self.Wq @ q + self.bi

    def score_one(self, p, h, q, level, diff):
        u = self.user_emb(p, h)
        i = self.item_emb(q)
        return float(u @ i - self.alpha * abs(int(level) - int(diff)))

    def fit(self, data: SyntheticData, epochs: int = 8, neg_per_pos: int = 2, verbose: bool = True):
        rng = np.random.default_rng(2025)
        train_triples = []

        n_items = data.Q.shape[0]
        all_items = np.arange(n_items)

        for u in data.train_users:
            pos_items = list(data.target_sets[u])
            blocked = data.seen_sets[u].union(data.target_sets[u])
            neg_pool = np.array([i for i in all_items if i not in blocked], dtype=int)
            if len(neg_pool) == 0:
                continue
            for ip in pos_items:
                for _ in range(neg_per_pos):
                    ineg = int(rng.choice(neg_pool))
                    train_triples.append((int(u), int(ip), int(ineg)))

        for epoch in range(epochs):
            rng.shuffle(train_triples)
            total_loss = 0.0
            for u, ip, ine in train_triples:
                p = data.P[u]
                h = data.H[u]
                qp = data.Q[ip]
                qn = data.Q[ine]
                lp = data.levels[u]
                dp = data.difficulty[ip]
                dn = data.difficulty[ine]

                eu = self.user_emb(p, h)
                eip = self.item_emb(qp)
                ein = self.item_emb(qn)

                sp = eu @ eip - self.alpha * abs(int(lp) - int(dp))
                sn = eu @ ein - self.alpha * abs(int(lp) - int(dn))
                x = sp - sn
                sig = sigmoid(x)
                loss = -np.log(sig + 1e-12)
                total_loss += float(loss)

                g = sig - 1.0  # d/dx of -log(sigmoid(x))

                d_eu = g * (eip - ein)
                d_eip = g * eu
                d_ein = -g * eu

                # gradient step
                self.Wp -= self.lr * (np.outer(d_eu, p) + self.reg * self.Wp)
                if self.use_history:
                    self.Wh -= self.lr * (np.outer(d_eu, h) + self.reg * self.Wh)
                self.Wq -= self.lr * (
                    np.outer(d_eip, qp) + np.outer(d_ein, qn) + self.reg * self.Wq
                )
                self.bu -= self.lr * d_eu
                self.bi -= self.lr * (d_eip + d_ein)

            if verbose:
                avg_loss = total_loss / max(len(train_triples), 1)
                print(f"Epoch {epoch+1:02d}/{epochs}  avg pairwise loss = {avg_loss:.5f}")

    def recommend(self, data: SyntheticData, candidate_users, n_recommend: int = 20):
        item_emb = data.Q @ self.Wq.T + self.bi[None, :]
        rankings = {}

        for u in candidate_users:
            eu = data.P[u] @ self.Wp.T + self.bu[None, :]
            if self.use_history:
                eu = eu + data.H[u] @ self.Wh.T
            eu = eu.reshape(-1)

            score = item_emb @ eu - self.alpha * np.abs(data.difficulty - data.levels[u])
            if data.seen_sets[u]:
                score[list(data.seen_sets[u])] = -1e18
            order = np.argsort(score)[::-1][:n_recommend]
            rankings[u] = order.tolist()
        return rankings


# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(
    output_dir: str,
    n_users: int = 600,
    n_items: int = 400,
    epochs: int = 8,
    latent_dim: int = 32,
    alpha_values = (0.0, 0.5, 1.0, 1.5, 2.0),
    tau: int = 2,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    data = generate_synthetic_benchmark(
        n_users=n_users,
        n_items=n_items,
        seed=seed,
    )

    print("Synthetic benchmark generated.")
    print(f"Users: {n_users}, Items: {n_items}, Train users: {len(data.train_users)}, Test users: {len(data.test_users)}")

    # ----- 5 methods -----
    rankings_random = recommend_random(data, data.test_users, n_recommend=20, seed=seed + 1)
    rankings_pop = recommend_popularity(data, data.test_users, n_recommend=20)
    rankings_cm = recommend_contentmatch(data, data.test_users, n_recommend=20)

    static_tt = LinearTwoTower(
        d_p=data.P.shape[1],
        d_h=data.H.shape[1],
        d_q=data.Q.shape[1],
        latent_dim=latent_dim,
        alpha=1.5,
        use_history=False,
        seed=seed + 2,
    )
    static_tt.fit(data, epochs=epochs, verbose=True)
    rankings_static = static_tt.recommend(data, data.test_users, n_recommend=20)

    dynamic_tt = LinearTwoTower(
        d_p=data.P.shape[1],
        d_h=data.H.shape[1],
        d_q=data.Q.shape[1],
        latent_dim=latent_dim,
        alpha=1.5,
        use_history=True,
        seed=seed + 3,
    )
    dynamic_tt.fit(data, epochs=epochs, verbose=True)
    rankings_dynamic = dynamic_tt.recommend(data, data.test_users, n_recommend=20)

    method_rankings = {
        "Random": rankings_random,
        "Popularity": rankings_pop,
        "ContentMatch": rankings_cm,
        "StaticTT": rankings_static,
        "DynamicTT": rankings_dynamic,
    }

    method_rows = []
    for method_name, rankings in method_rankings.items():
        metric_df = evaluate_rankings(
            rankings,
            data.target_sets,
            data.levels,
            data.difficulty,
            tau=tau,
            ks=(5, 10, 20),
        )
        for _, row in metric_df.iterrows():
            method_rows.append({
                "Method": method_name,
                "K": int(row["K"]),
                "HitRate": row["HitRate"],
                "Recall": row["Recall"],
                "NDCG": row["NDCG"],
                "DMR": row["DMR"],
            })
    method_df = pd.DataFrame(method_rows)
    method_df.to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)

    # ----- 5 alpha values with DynamicTT -----
    alpha_rows = []
    for alpha in alpha_values:
        model = LinearTwoTower(
            d_p=data.P.shape[1],
            d_h=data.H.shape[1],
            d_q=data.Q.shape[1],
            latent_dim=latent_dim,
            alpha=float(alpha),
            use_history=True,
            seed=seed + 100 + int(alpha * 10),
        )
        print(f"\nTraining DynamicTT for alpha={alpha:.2f}")
        model.fit(data, epochs=epochs, verbose=False)
        rankings = model.recommend(data, data.test_users, n_recommend=20)
        metric_df = evaluate_rankings(
            rankings,
            data.target_sets,
            data.levels,
            data.difficulty,
            tau=tau,
            ks=(10,),
        )
        row = metric_df.iloc[0].to_dict()
        row["alpha"] = float(alpha)
        alpha_rows.append(row)

    alpha_df = pd.DataFrame(alpha_rows)[["alpha", "HitRate", "Recall", "NDCG", "DMR"]]
    alpha_df.to_csv(os.path.join(output_dir, "alpha_sensitivity.csv"), index=False)

    # ----- plots -----
    plot_method_comparison(method_df, output_dir)
    plot_alpha_sensitivity(alpha_df, output_dir)

    # ----- summary -----
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Method comparison at K=10\n")
        f.write(method_df[method_df["K"] == 10].sort_values("Recall", ascending=False).to_string(index=False))
        f.write("\n\nAlpha sensitivity (DynamicTT at K=10)\n")
        f.write(alpha_df.sort_values("alpha").to_string(index=False))
        f.write("\n")

    return method_df, alpha_df


def plot_method_comparison(method_df: pd.DataFrame, output_dir: str):
    df = method_df[method_df["K"] == 10].copy()
    methods = df["Method"].tolist()

    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(methods))
    width = 0.2
    plt.bar(x - 1.5 * width, df["HitRate"], width, label="HitRate@10")
    plt.bar(x - 0.5 * width, df["Recall"], width, label="Recall@10")
    plt.bar(x + 0.5 * width, df["NDCG"], width, label="NDCG@10")
    plt.bar(x + 1.5 * width, df["DMR"], width, label="DMR@10")
    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Metric value")
    plt.title("Five-method comparison at K=10")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_method_comparison.png"), dpi=200)
    plt.close()


def plot_alpha_sensitivity(alpha_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_df["alpha"], alpha_df["Recall"], marker="o", label="Recall@10")
    plt.plot(alpha_df["alpha"], alpha_df["NDCG"], marker="s", label="NDCG@10")
    plt.plot(alpha_df["alpha"], alpha_df["DMR"], marker="^", label="DMR@10")
    plt.xlabel("Difficulty coefficient alpha")
    plt.ylabel("Metric value")
    plt.title("Sensitivity of DynamicTT to alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_alpha_sensitivity.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dynamic two-tower synthetic benchmark.")
    parser.add_argument("--output_dir", type=str, default="python_results", help="Directory for csv/png/txt outputs.")
    parser.add_argument("--n_users", type=int, default=600, help="Number of synthetic users. Use 1200 for the paper-scale setting.")
    parser.add_argument("--n_items", type=int, default=400, help="Number of synthetic items. Use 800 for the paper-scale setting.")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs for two-tower models. Increase for the full paper-scale run.")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension of two-tower models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    method_df, alpha_df = run_experiment(
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )

    print("\n=== Method comparison (K=10) ===")
    print(method_df[method_df["K"] == 10].sort_values("Recall", ascending=False).to_string(index=False))
    print("\n=== Alpha sensitivity (K=10) ===")
    print(alpha_df.sort_values("alpha").to_string(index=False))
    print(f"\nFiles saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
