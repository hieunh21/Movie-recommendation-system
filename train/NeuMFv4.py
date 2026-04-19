# Cell 1: imports + config

import gc
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers


PAPER_TABLE2_MOVIELENS = {
    8:  {"HR@10": 0.684, "NDCG@10": 0.403},
    16: {"HR@10": 0.707, "NDCG@10": 0.426},
    32: {"HR@10": 0.726, "NDCG@10": 0.445},
    64: {"HR@10": 0.730, "NDCG@10": 0.447},
}


@dataclass
class RunConfig:
    ratings_path: Optional[str] = None
    output_dir: str = "/kaggle/working/neumf_paper_notebook"

    seed: int = 42
    require_gpu: bool = True
    use_all_gpus: bool = False   # default False for stability; bật True nếu bạn chắc chắn môi trường đủ ổn định
    mixed_precision: bool = False

    topk: int = 10
    train_num_neg: int = 4
    eval_num_neg: int = 100          # paper PDF
    use_official_repo_99_eval_neg: bool = False

    batch_size: int = 256            # paper / official repo examples
    adam_lr: float = 1e-3            # paper / official repo examples
    sgd_lr: float = 1e-2             # paper-faithful NeuMF pretraining optimizer choice

    pretrain_epochs: int = 10        # paper says the most effective updates are in first 10 iterations
    neumf_epochs: int = 10
    alpha: float = 0.5

    factors_grid: Tuple[int, ...] = (8, 16, 32, 64)

    run_without_pretraining: bool = True
    run_with_pretraining: bool = True

    save_best_weights: bool = True
    verbose: int = 1

    @property
    def effective_eval_num_neg(self) -> int:
        return 99 if self.use_official_repo_99_eval_neg else self.eval_num_neg


cfg = RunConfig()
pd.DataFrame([asdict(cfg)])

# %%

# Cell 2: reproducibility + device setup

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def setup_tensorflow(
    require_gpu: bool = True,
    use_all_gpus: bool = False,
    mixed_precision_enabled: bool = False,
):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Detected GPUs: {len(gpus)} | CPUs: {len(cpus)}")

    if require_gpu and not gpus:
        raise RuntimeError("Không tìm thấy GPU nhưng cfg.require_gpu=True")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if mixed_precision_enabled:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
    else:
        keras.mixed_precision.set_global_policy("float32")
        print("Mixed precision disabled.")

    if len(gpus) > 1 and use_all_gpus:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    logical_gpus = tf.config.list_logical_devices("GPU")
    logical_names = [d.name for d in logical_gpus] if logical_gpus else []
    print("Visible logical GPUs:", logical_names)
    print("Replicas in sync:", strategy.num_replicas_in_sync)
    return strategy


set_global_seed(cfg.seed)
STRATEGY = setup_tensorflow(
    require_gpu=cfg.require_gpu,
    use_all_gpus=cfg.use_all_gpus,
    mixed_precision_enabled=cfg.mixed_precision,
)

# %%

# Cell 3: data loading

def autodetect_ratings_path(root: str = "/kaggle/input") -> str:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input root không tồn tại: {root}")

    preferred_names = ["ratings.dat", "ratings.csv"]
    candidates = []
    for name in preferred_names:
        candidates.extend(root_path.rglob(name))

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy ratings.dat hoặc ratings.csv dưới /kaggle/input. "
            "Hãy set cfg.ratings_path thủ công."
        )

    def sort_key(p: Path):
        s = str(p).lower()
        score = 0
        if s.endswith("ratings.dat"):
            score -= 2
        if "ml-1m" in s:
            score -= 2
        if "movielens" in s:
            score -= 1
        return (score, len(p.parts), s)

    return str(sorted(candidates, key=sort_key)[0])


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in {"userid", "user_id", "user", "uid"}:
            renamed[c] = "user_id"
        elif key in {"movieid", "itemid", "item_id", "item", "iid"}:
            renamed[c] = "item_id"
        elif key in {"rating", "ratings"}:
            renamed[c] = "rating"
        elif key in {"timestamp", "time", "ts"}:
            renamed[c] = "timestamp"

    df = df.rename(columns=renamed)

    if list(df.columns[:4]) == [0, 1, 2, 3]:
        df.columns = ["user_id", "item_id", "rating", "timestamp"]

    required = {"user_id", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    if "rating" not in df.columns:
        df["rating"] = 1
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df), dtype=np.int64)

    return df[["user_id", "item_id", "rating", "timestamp"]].copy()


def load_ratings_frame(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file ratings: {path}")

    head = path_obj.read_text(encoding="utf-8", errors="ignore")[:5000]

    if "::" in head:
        df = pd.read_csv(
            path_obj,
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
    else:
        sep = None
        for cand in [",", ";", "\t", "|"]:
            try:
                test_df = pd.read_csv(path_obj, sep=cand, nrows=5)
                if test_df.shape[1] >= 3:
                    sep = cand
                    break
            except Exception:
                pass
        if sep is None:
            raise ValueError("Không suy ra được delimiter cho ratings file.")
        df = pd.read_csv(path_obj, sep=sep)
        df = _normalize_columns(df)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)

    # implicit feedback: mọi interaction quan sát được xem là positive
    # nếu trùng (user, item), giữ interaction mới nhất
    df = (
        df.sort_values(["user_id", "item_id", "timestamp"])
          .drop_duplicates(["user_id", "item_id"], keep="last")
          .sort_values(["user_id", "timestamp", "item_id"])
          .reset_index(drop=True)
    )

    user_codes = {u: idx for idx, u in enumerate(df["user_id"].drop_duplicates().tolist())}
    item_codes = {i: idx for idx, i in enumerate(df["item_id"].drop_duplicates().tolist())}

    df["user_idx"] = df["user_id"].map(user_codes).astype(np.int32)
    df["item_idx"] = df["item_id"].map(item_codes).astype(np.int32)

    return df[["user_idx", "item_idx", "timestamp"]].copy()


if cfg.ratings_path is None:
    cfg.ratings_path = autodetect_ratings_path("/kaggle/input")

ratings_df = load_ratings_frame(cfg.ratings_path)
num_users = int(ratings_df["user_idx"].nunique())
num_items = int(ratings_df["item_idx"].nunique())

print("ratings_path:", cfg.ratings_path)
print("Raw interactions:", f"{len(ratings_df):,}")
print(f"num_users={num_users:,} | num_items={num_items:,}")
ratings_df.head()

# %%

# Cell 4: split + negative samples

@dataclass
class PreparedData:
    num_users: int
    num_items: int
    train_pairs: np.ndarray
    val_ratings: np.ndarray
    test_ratings: np.ndarray
    val_negatives: List[np.ndarray]
    test_negatives: List[np.ndarray]
    user_all_positives: Dict[int, set]
    user_train_positives: Dict[int, set]


def sample_negatives_excluding(
    rng: np.random.Generator,
    positives: set,
    num_items: int,
    num_neg: int,
) -> np.ndarray:
    sampled = []
    sampled_set = set()
    while len(sampled) < num_neg:
        candidates = rng.integers(0, num_items, size=max(64, num_neg * 4))
        for j in candidates.tolist():
            if j not in positives and j not in sampled_set:
                sampled.append(int(j))
                sampled_set.add(int(j))
                if len(sampled) == num_neg:
                    break
    return np.asarray(sampled, dtype=np.int32)


def prepare_leave_one_out_with_random_validation(
    ratings: pd.DataFrame,
    num_users: int,
    num_items: int,
    eval_num_neg: int,
    seed: int,
) -> PreparedData:
    rng = np.random.default_rng(seed)

    train_pairs: List[Tuple[int, int]] = []
    val_ratings: List[Tuple[int, int]] = []
    test_ratings: List[Tuple[int, int]] = []

    user_all_positives: Dict[int, set] = {}
    user_train_positives: Dict[int, set] = {}

    grouped = ratings.groupby("user_idx", sort=True)
    for user, user_df in grouped:
        items = user_df.sort_values(["timestamp", "item_idx"])["item_idx"].tolist()
        if len(items) < 3:
            raise ValueError(
                f"User {user} có {len(items)} interactions. "
                "Cần >= 3 để có train/val/test theo protocol này."
            )

        test_item = int(items[-1])  # latest interaction -> test
        val_pick = int(rng.integers(0, len(items) - 1))  # random one from remaining -> validation
        val_item = int(items[val_pick])

        train_items = [int(it) for idx, it in enumerate(items) if idx != len(items) - 1 and idx != val_pick]
        if not train_items:
            raise ValueError(f"User {user} không còn train item sau split.")

        user_all_positives[int(user)] = set(map(int, items))
        user_train_positives[int(user)] = set(train_items)

        train_pairs.extend((int(user), int(it)) for it in train_items)
        val_ratings.append((int(user), val_item))
        test_ratings.append((int(user), test_item))

    val_negatives = []
    test_negatives = []

    rng_val = np.random.default_rng(seed + 1000)
    rng_test = np.random.default_rng(seed + 2000)

    for user in range(num_users):
        positives = user_all_positives[user]
        val_negatives.append(
            sample_negatives_excluding(rng_val, positives, num_items, eval_num_neg)
        )
        test_negatives.append(
            sample_negatives_excluding(rng_test, positives, num_items, eval_num_neg)
        )

    return PreparedData(
        num_users=num_users,
        num_items=num_items,
        train_pairs=np.asarray(train_pairs, dtype=np.int32),
        val_ratings=np.asarray(val_ratings, dtype=np.int32),
        test_ratings=np.asarray(test_ratings, dtype=np.int32),
        val_negatives=val_negatives,
        test_negatives=test_negatives,
        user_all_positives=user_all_positives,
        user_train_positives=user_train_positives,
    )


def audit_prepared_data(prepared: PreparedData, eval_num_neg: int) -> None:
    assert prepared.val_ratings.shape[0] == prepared.num_users
    assert prepared.test_ratings.shape[0] == prepared.num_users
    assert len(prepared.val_negatives) == prepared.num_users
    assert len(prepared.test_negatives) == prepared.num_users

    for u in range(prepared.num_users):
        val_item = int(prepared.val_ratings[u, 1])
        test_item = int(prepared.test_ratings[u, 1])

        assert val_item not in prepared.user_train_positives[u]
        assert test_item not in prepared.user_train_positives[u]
        assert val_item in prepared.user_all_positives[u]
        assert test_item in prepared.user_all_positives[u]

        vnegs = prepared.val_negatives[u]
        tnegs = prepared.test_negatives[u]

        assert len(vnegs) == eval_num_neg
        assert len(tnegs) == eval_num_neg
        assert len(np.unique(vnegs)) == eval_num_neg
        assert len(np.unique(tnegs)) == eval_num_neg

        assert not (set(vnegs.tolist()) & prepared.user_all_positives[u])
        assert not (set(tnegs.tolist()) & prepared.user_all_positives[u])


prepared = prepare_leave_one_out_with_random_validation(
    ratings=ratings_df,
    num_users=num_users,
    num_items=num_items,
    eval_num_neg=cfg.effective_eval_num_neg,
    seed=cfg.seed,
)
audit_prepared_data(prepared, cfg.effective_eval_num_neg)

print("Protocol audit passed.")
print("train =", f"{len(prepared.train_pairs):,}",
      "| val =", f"{len(prepared.val_ratings):,}",
      "| test =", f"{len(prepared.test_ratings):,}")

# %%

# Cell 5: model definitions (paper-faithful)

def normal_init():
    return initializers.RandomNormal(mean=0.0, stddev=0.01)


def paper_mlp_embedding_dim(factors: int) -> int:
    # paper sentence:
    # factors=8 => neural CF layers 32 -> 16 -> 8, embedding size = 16
    return factors * 2


def paper_dense_layers(factors: int) -> List[int]:
    return [factors * 4, factors * 2, factors]


def build_gmf(num_users: int, num_items: int, factors: int) -> keras.Model:
    user_in = keras.Input(shape=(1,), dtype="int32", name="user")
    item_in = keras.Input(shape=(1,), dtype="int32", name="item")

    user_emb = layers.Embedding(
        input_dim=num_users,
        output_dim=factors,
        embeddings_initializer=normal_init(),
        name="user_embedding",
    )(user_in)
    item_emb = layers.Embedding(
        input_dim=num_items,
        output_dim=factors,
        embeddings_initializer=normal_init(),
        name="item_embedding",
    )(item_in)

    user_vec = layers.Flatten(name="user_flat")(user_emb)
    item_vec = layers.Flatten(name="item_flat")(item_emb)
    x = layers.Multiply(name="gmf_interaction")([user_vec, item_vec])

    out = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=normal_init(),
        bias_initializer="zeros",
        name="prediction",
    )(x)

    return keras.Model([user_in, item_in], out, name=f"GMF_f{factors}")


def build_mlp(num_users: int, num_items: int, factors: int) -> keras.Model:
    emb_dim = paper_mlp_embedding_dim(factors)
    dense_sizes = paper_dense_layers(factors)

    user_in = keras.Input(shape=(1,), dtype="int32", name="user")
    item_in = keras.Input(shape=(1,), dtype="int32", name="item")

    user_emb = layers.Embedding(
        input_dim=num_users,
        output_dim=emb_dim,
        embeddings_initializer=normal_init(),
        name="user_embedding",
    )(user_in)
    item_emb = layers.Embedding(
        input_dim=num_items,
        output_dim=emb_dim,
        embeddings_initializer=normal_init(),
        name="item_embedding",
    )(item_in)

    user_vec = layers.Flatten(name="user_flat")(user_emb)
    item_vec = layers.Flatten(name="item_flat")(item_emb)
    x = layers.Concatenate(name="concat")([user_vec, item_vec])

    for idx, units in enumerate(dense_sizes, start=1):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_initializer=normal_init(),
            bias_initializer="zeros",
            name=f"layer{idx}",
        )(x)

    out = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=normal_init(),
        bias_initializer="zeros",
        name="prediction",
    )(x)

    return keras.Model([user_in, item_in], out, name=f"MLP_f{factors}")


def build_neumf(num_users: int, num_items: int, factors: int) -> keras.Model:
    emb_dim = paper_mlp_embedding_dim(factors)
    dense_sizes = paper_dense_layers(factors)

    user_in = keras.Input(shape=(1,), dtype="int32", name="user")
    item_in = keras.Input(shape=(1,), dtype="int32", name="item")

    mf_user_emb = layers.Embedding(
        input_dim=num_users,
        output_dim=factors,
        embeddings_initializer=normal_init(),
        name="mf_embedding_user",
    )(user_in)
    mf_item_emb = layers.Embedding(
        input_dim=num_items,
        output_dim=factors,
        embeddings_initializer=normal_init(),
        name="mf_embedding_item",
    )(item_in)

    mlp_user_emb = layers.Embedding(
        input_dim=num_users,
        output_dim=emb_dim,
        embeddings_initializer=normal_init(),
        name="mlp_embedding_user",
    )(user_in)
    mlp_item_emb = layers.Embedding(
        input_dim=num_items,
        output_dim=emb_dim,
        embeddings_initializer=normal_init(),
        name="mlp_embedding_item",
    )(item_in)

    mf_user_vec = layers.Flatten(name="mf_user_flat")(mf_user_emb)
    mf_item_vec = layers.Flatten(name="mf_item_flat")(mf_item_emb)
    mf_vector = layers.Multiply(name="mf_interaction")([mf_user_vec, mf_item_vec])

    mlp_user_vec = layers.Flatten(name="mlp_user_flat")(mlp_user_emb)
    mlp_item_vec = layers.Flatten(name="mlp_item_flat")(mlp_item_emb)
    mlp_vector = layers.Concatenate(name="mlp_concat")([mlp_user_vec, mlp_item_vec])

    for idx, units in enumerate(dense_sizes, start=1):
        mlp_vector = layers.Dense(
            units,
            activation="relu",
            kernel_initializer=normal_init(),
            bias_initializer="zeros",
            name=f"layer{idx}",
        )(mlp_vector)

    predict_vector = layers.Concatenate(name="predict_vector")([mf_vector, mlp_vector])

    out = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=normal_init(),
        bias_initializer="zeros",
        name="prediction",
    )(predict_vector)

    return keras.Model([user_in, item_in], out, name=f"NeuMF_f{factors}")


for f in cfg.factors_grid:
    print(f"factors={f} | mlp_embedding_dim={paper_mlp_embedding_dim(f)} | dense_layers={paper_dense_layers(f)}")

# %%

# Cell 6: train arrays + evaluation

def make_training_arrays(
    train_pairs: np.ndarray,
    user_all_positives: Dict[int, set],
    num_items: int,
    num_neg: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    users: List[int] = []
    items: List[int] = []
    labels: List[float] = []

    for u, i in train_pairs:
        u = int(u)
        i = int(i)

        users.append(u)
        items.append(i)
        labels.append(1.0)

        positives = user_all_positives[u]
        sampled = []
        sampled_set = set()

        while len(sampled) < num_neg:
            candidates = rng.integers(0, num_items, size=max(16, num_neg * 4))
            for j in candidates.tolist():
                if j not in positives and j not in sampled_set:
                    sampled.append(int(j))
                    sampled_set.add(int(j))
                    if len(sampled) == num_neg:
                        break

        for j in sampled:
            users.append(u)
            items.append(j)
            labels.append(0.0)

    users = np.asarray(users, dtype=np.int32)
    items = np.asarray(items, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.float32)

    order = np.arange(len(labels))
    rng.shuffle(order)
    return users[order], items[order], labels[order]


def build_eval_arrays(ratings: np.ndarray, negatives: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    users = []
    items = []
    for idx in range(len(ratings)):
        u, pos = ratings[idx]
        candidates = np.concatenate([[pos], negatives[idx]]).astype(np.int32)
        users.extend([int(u)] * len(candidates))
        items.extend(candidates.tolist())
    return np.asarray(users, dtype=np.int32), np.asarray(items, dtype=np.int32)


def _rank_metrics_from_row(scores: np.ndarray, topk: int) -> Tuple[float, float]:
    # candidate[0] luôn là positive item
    order = np.argsort(-scores)
    rank = int(np.where(order == 0)[0][0])  # 0-based
    hr = 1.0 if rank < topk else 0.0
    ndcg = float(1.0 / np.log2(rank + 2.0)) if rank < topk else 0.0
    return hr, ndcg


def evaluate_model(
    model: keras.Model,
    ratings: np.ndarray,
    negatives: List[np.ndarray],
    topk: int,
    batch_size: int = 65536,
) -> Dict[str, float]:
    eval_users, eval_items = build_eval_arrays(ratings, negatives)
    num_candidates = 1 + len(negatives[0])

    preds = model.predict(
        [eval_users, eval_items],
        batch_size=batch_size,
        verbose=0,
    ).reshape(len(ratings), num_candidates)

    hrs = []
    ndcgs = []

    for row in preds:
        hr, ndcg = _rank_metrics_from_row(row, topk)
        hrs.append(hr)
        ndcgs.append(ndcg)

    return {
        "HR@10": float(np.mean(hrs)),
        "NDCG@10": float(np.mean(ndcgs)),
    }

# %%

# Cell 7: training helpers

@dataclass
class TrainResult:
    name: str
    factors: int
    optimizer: str
    lr: float
    best_epoch: int
    best_val_hr: float
    best_val_ndcg: float
    test_hr: float
    test_ndcg: float
    weight_path: Optional[str]
    history: List[Dict[str, float]]


def make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def compile_model(model: keras.Model, optimizer_name: str, lr: float) -> None:
    model.compile(
        optimizer=make_optimizer(optimizer_name, lr),
        loss="binary_crossentropy",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def better_score(
    hr: float,
    ndcg: float,
    best_hr: float,
    best_ndcg: float,
) -> bool:
    if hr > best_hr:
        return True
    if hr == best_hr and ndcg > best_ndcg:
        return True
    return False


def train_single_model(
    model: keras.Model,
    model_name: str,
    prepared: PreparedData,
    output_dir: Path,
    factors: int,
    optimizer_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    topk: int,
    save_best_weights: bool = True,
) -> TrainResult:
    ensure_dir(output_dir)
    compile_model(model, optimizer_name, lr)

    best_epoch = -1
    best_val_hr = -1.0
    best_val_ndcg = -1.0
    weight_path = output_dir / f"{model_name}_best.weights.h5"

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_users, train_items, train_labels = make_training_arrays(
            train_pairs=prepared.train_pairs,
            user_all_positives=prepared.user_all_positives,
            num_items=prepared.num_items,
            num_neg=cfg.train_num_neg,
            seed=seed + epoch,
        )

        hist = model.fit(
            [train_users, train_items],
            train_labels,
            batch_size=batch_size,
            epochs=1,
            shuffle=False,  # arrays đã được shuffle ở trên
            verbose=0,
        )

        train_loss = float(hist.history["loss"][0])

        val_metrics = evaluate_model(
            model=model,
            ratings=prepared.val_ratings,
            negatives=prepared.val_negatives,
            topk=topk,
        )

        row = {
            "epoch": epoch,
            "loss": train_loss,
            "val_HR@10": val_metrics["HR@10"],
            "val_NDCG@10": val_metrics["NDCG@10"],
        }
        history.append(row)

        if cfg.verbose:
            print(
                f"[{model_name}] epoch={epoch:03d} "
                f"loss={train_loss:.6f} "
                f"HR@10={val_metrics['HR@10']:.6f} "
                f"NDCG@10={val_metrics['NDCG@10']:.6f}"
            )

        if better_score(
            val_metrics["HR@10"],
            val_metrics["NDCG@10"],
            best_val_hr,
            best_val_ndcg,
        ):
            best_epoch = epoch
            best_val_hr = val_metrics["HR@10"]
            best_val_ndcg = val_metrics["NDCG@10"]
            if save_best_weights:
                model.save_weights(weight_path)

    if save_best_weights and weight_path.exists():
        model.load_weights(weight_path)

    test_metrics = evaluate_model(
        model=model,
        ratings=prepared.test_ratings,
        negatives=prepared.test_negatives,
        topk=topk,
    )

    return TrainResult(
        name=model_name,
        factors=factors,
        optimizer=optimizer_name,
        lr=lr,
        best_epoch=best_epoch,
        best_val_hr=best_val_hr,
        best_val_ndcg=best_val_ndcg,
        test_hr=test_metrics["HR@10"],
        test_ndcg=test_metrics["NDCG@10"],
        weight_path=str(weight_path) if save_best_weights else None,
        history=history,
    )

# %%

# Cell 8: pretraining loader + experiment runners

def load_pretrained_weights(
    neumf: keras.Model,
    gmf: keras.Model,
    mlp: keras.Model,
    alpha: float = 0.5,
) -> keras.Model:
    # MF embeddings
    neumf.get_layer("mf_embedding_user").set_weights(gmf.get_layer("user_embedding").get_weights())
    neumf.get_layer("mf_embedding_item").set_weights(gmf.get_layer("item_embedding").get_weights())

    # MLP embeddings
    neumf.get_layer("mlp_embedding_user").set_weights(mlp.get_layer("user_embedding").get_weights())
    neumf.get_layer("mlp_embedding_item").set_weights(mlp.get_layer("item_embedding").get_weights())

    # MLP hidden layers
    factors = int(gmf.get_layer("user_embedding").output.shape[-1])
    num_layers = len(paper_dense_layers(factors))
    for i in range(1, num_layers + 1):
        neumf.get_layer(f"layer{i}").set_weights(mlp.get_layer(f"layer{i}").get_weights())

    # Prediction layer
    gmf_kernel, gmf_bias = gmf.get_layer("prediction").get_weights()
    mlp_kernel, mlp_bias = mlp.get_layer("prediction").get_weights()

    new_kernel = np.concatenate(
        [alpha * gmf_kernel, (1.0 - alpha) * mlp_kernel],
        axis=0,
    )
    new_bias = alpha * gmf_bias + (1.0 - alpha) * mlp_bias
    neumf.get_layer("prediction").set_weights([new_kernel, new_bias])

    return neumf


def clear_keras_state():
    keras.backend.clear_session()
    gc.collect()


def build_model_in_scope(kind: str, factors: int) -> keras.Model:
    with STRATEGY.scope():
        if kind == "gmf":
            return build_gmf(prepared.num_users, prepared.num_items, factors)
        if kind == "mlp":
            return build_mlp(prepared.num_users, prepared.num_items, factors)
        if kind == "neumf":
            return build_neumf(prepared.num_users, prepared.num_items, factors)
    raise ValueError(kind)


def run_neumf_without_pretraining(factors: int, root: Path) -> TrainResult:
    clear_keras_state()
    model = build_model_in_scope("neumf", factors)

    result = train_single_model(
        model=model,
        model_name=f"NeuMF-scratch-f{factors}",
        prepared=prepared,
        output_dir=root / f"f{factors}" / "without_pretraining",
        factors=factors,
        optimizer_name="adam",
        lr=cfg.adam_lr,
        epochs=cfg.neumf_epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed + 1000 * factors,
        topk=cfg.topk,
        save_best_weights=cfg.save_best_weights,
    )
    return result


def run_neumf_with_pretraining(factors: int, root: Path) -> Dict[str, TrainResult]:
    model_root = root / f"f{factors}" / "with_pretraining"
    ensure_dir(model_root)

    clear_keras_state()
    gmf = build_model_in_scope("gmf", factors)
    gmf_result = train_single_model(
        model=gmf,
        model_name=f"GMF-pretrain-f{factors}",
        prepared=prepared,
        output_dir=model_root / "gmf",
        factors=factors,
        optimizer_name="adam",
        lr=cfg.adam_lr,
        epochs=cfg.pretrain_epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed + 10 * factors,
        topk=cfg.topk,
        save_best_weights=cfg.save_best_weights,
    )

    clear_keras_state()
    mlp = build_model_in_scope("mlp", factors)
    mlp_result = train_single_model(
        model=mlp,
        model_name=f"MLP-pretrain-f{factors}",
        prepared=prepared,
        output_dir=model_root / "mlp",
        factors=factors,
        optimizer_name="adam",
        lr=cfg.adam_lr,
        epochs=cfg.pretrain_epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed + 20 * factors,
        topk=cfg.topk,
        save_best_weights=cfg.save_best_weights,
    )

    clear_keras_state()
    best_gmf = build_model_in_scope("gmf", factors)
    best_mlp = build_model_in_scope("mlp", factors)
    best_gmf.load_weights(gmf_result.weight_path)
    best_mlp.load_weights(mlp_result.weight_path)

    clear_keras_state()
    neumf = build_model_in_scope("neumf", factors)
    neumf = load_pretrained_weights(neumf, best_gmf, best_mlp, alpha=cfg.alpha)

    neumf_result = train_single_model(
        model=neumf,
        model_name=f"NeuMF-pretrained-f{factors}",
        prepared=prepared,
        output_dir=model_root / "neumf",
        factors=factors,
        optimizer_name="sgd",
        lr=cfg.sgd_lr,
        epochs=cfg.neumf_epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed + 30 * factors,
        topk=cfg.topk,
        save_best_weights=cfg.save_best_weights,
    )

    return {
        "gmf": gmf_result,
        "mlp": mlp_result,
        "neumf": neumf_result,
    }

# %%

# Cell 9: main experiment loop (Table 2 style)

root = Path(cfg.output_dir)
ensure_dir(root)

rows = []

for factors in cfg.factors_grid:
    print("=" * 100)
    print(f"Running factors={factors}")

    if cfg.run_without_pretraining:
        scratch_result = run_neumf_without_pretraining(factors, root)
        rows.append({
            "factors": factors,
            "variant": "without_pretraining",
            "optimizer": scratch_result.optimizer,
            "lr": scratch_result.lr,
            "best_epoch": scratch_result.best_epoch,
            "val_HR@10": scratch_result.best_val_hr,
            "val_NDCG@10": scratch_result.best_val_ndcg,
            "test_HR@10": scratch_result.test_hr,
            "test_NDCG@10": scratch_result.test_ndcg,
            "paper_HR@10": PAPER_TABLE2_MOVIELENS[factors]["HR@10"],
            "paper_NDCG@10": PAPER_TABLE2_MOVIELENS[factors]["NDCG@10"],
            "delta_HR_vs_paper_pretrained": scratch_result.test_hr - PAPER_TABLE2_MOVIELENS[factors]["HR@10"],
            "delta_NDCG_vs_paper_pretrained": scratch_result.test_ndcg - PAPER_TABLE2_MOVIELENS[factors]["NDCG@10"],
            "weights": scratch_result.weight_path,
        })
        pd.DataFrame(rows).to_csv(root / "results_interim.csv", index=False)

    if cfg.run_with_pretraining:
        pretrain_results = run_neumf_with_pretraining(factors, root)
        neumf_pre = pretrain_results["neumf"]

        rows.append({
            "factors": factors,
            "variant": "with_pretraining",
            "optimizer": neumf_pre.optimizer,
            "lr": neumf_pre.lr,
            "best_epoch": neumf_pre.best_epoch,
            "val_HR@10": neumf_pre.best_val_hr,
            "val_NDCG@10": neumf_pre.best_val_ndcg,
            "test_HR@10": neumf_pre.test_hr,
            "test_NDCG@10": neumf_pre.test_ndcg,
            "paper_HR@10": PAPER_TABLE2_MOVIELENS[factors]["HR@10"],
            "paper_NDCG@10": PAPER_TABLE2_MOVIELENS[factors]["NDCG@10"],
            "delta_HR_vs_paper_pretrained": neumf_pre.test_hr - PAPER_TABLE2_MOVIELENS[factors]["HR@10"],
            "delta_NDCG_vs_paper_pretrained": neumf_pre.test_ndcg - PAPER_TABLE2_MOVIELENS[factors]["NDCG@10"],
            "weights": neumf_pre.weight_path,
        })
        pd.DataFrame(rows).to_csv(root / "results_interim.csv", index=False)

results_df = pd.DataFrame(rows)
results_df

# %%

# Cell 10: summary tables

if results_df.empty:
    raise ValueError("results_df is empty")

summary_table = (
    results_df.pivot(
        index="factors",
        columns="variant",
        values=["test_HR@10", "test_NDCG@10"]
    )
    .sort_index()
)

paper_table = pd.DataFrame(
    [
        {
            "factors": f,
            "paper_pretrained_HR@10": vals["HR@10"],
            "paper_pretrained_NDCG@10": vals["NDCG@10"],
        }
        for f, vals in PAPER_TABLE2_MOVIELENS.items()
    ]
).sort_values("factors")

compare_pretrained = (
    results_df[results_df["variant"] == "with_pretraining"]
    [["factors", "test_HR@10", "test_NDCG@10"]]
    .rename(columns={
        "test_HR@10": "our_pretrained_HR@10",
        "test_NDCG@10": "our_pretrained_NDCG@10",
    })
    .merge(paper_table, on="factors", how="right")
    .sort_values("factors")
)

compare_pretrained["delta_HR"] = compare_pretrained["our_pretrained_HR@10"] - compare_pretrained["paper_pretrained_HR@10"]
compare_pretrained["delta_NDCG"] = compare_pretrained["our_pretrained_NDCG@10"] - compare_pretrained["paper_pretrained_NDCG@10"]

results_df.to_csv(root / "results_final.csv", index=False)
summary_table.to_csv(root / "summary_table.csv")
compare_pretrained.to_csv(root / "compare_pretrained_vs_paper.csv", index=False)

print("Saved:")
print(root / "results_final.csv")
print(root / "summary_table.csv")
print(root / "compare_pretrained_vs_paper.csv")

try:
    from IPython.display import display
    display(summary_table)
    display(compare_pretrained)
except Exception:
    print(summary_table)
    print(compare_pretrained)

# %%

# Cell 11: export full models + load models for inference

def export_full_models(results_df: pd.DataFrame, root: Path) -> pd.DataFrame:
    export_rows = []
    export_root = root / "exported_models"
    ensure_dir(export_root)

    for _, row in results_df.iterrows():
        variant = str(row["variant"])
        factors = int(row["factors"])
        weights_path = row.get("weights")

        if not weights_path or pd.isna(weights_path):
            continue

        clear_keras_state()
        model = build_model_in_scope("neumf", factors)
        model.load_weights(str(weights_path))

        model_dir = export_root / f"f{factors}"
        ensure_dir(model_dir)
        model_path = model_dir / f"{variant}.keras"
        model.save(model_path)

        export_rows.append({
            "factors": factors,
            "variant": variant,
            "weights_path": str(weights_path),
            "model_path": str(model_path),
        })

        del model
        clear_keras_state()

    exported_df = pd.DataFrame(export_rows).sort_values(["factors", "variant"]).reset_index(drop=True)
    exported_df.to_csv(root / "exported_models.csv", index=False)
    return exported_df


def load_saved_model(model_path: str, compile_model: bool = False) -> keras.Model:
    return keras.models.load_model(model_path, compile=compile_model)


exported_models_df = export_full_models(results_df, root)
print("Saved:")
print(root / "exported_models.csv")
try:
    from IPython.display import display
    display(exported_models_df)
except Exception:
    print(exported_models_df)

# example: load best pretrained model and score one (user, item)
pretrained_only = exported_models_df[exported_models_df["variant"] == "with_pretraining"].copy()
if not pretrained_only.empty:
    best_pretrained_result = (
        results_df[results_df["variant"] == "with_pretraining"]
        .sort_values(["test_HR@10", "test_NDCG@10"], ascending=False)
        .iloc[0]
    )
    best_factors = int(best_pretrained_result["factors"])
    best_model_path = pretrained_only.loc[
        pretrained_only["factors"] == best_factors, "model_path"
    ].iloc[0]

    loaded_model = load_saved_model(best_model_path, compile_model=False)
    sample_user = np.asarray([int(prepared.test_ratings[0, 0])], dtype=np.int32)
    sample_item = np.asarray([int(prepared.test_ratings[0, 1])], dtype=np.int32)
    sample_score = float(loaded_model.predict([sample_user, sample_item], verbose=0)[0, 0])

    print("Best pretrained model:", best_model_path)
    print("Sample inference score:", sample_score)
