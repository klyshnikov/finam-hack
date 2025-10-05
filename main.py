from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import re, ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

candles = pd.read_csv("candles.csv")

try:
    from IPython.display import display
    display(candles.head())
except Exception:
    print(candles.head().to_string())

tickers = candles["ticker"].unique().tolist()

ticker_patterns = {
    "AFLT": [r"(?:^|[^а-яa-zё])аэрофлот[а-яё]*", r"(?:^|[^а-яa-zё])aeroflot[a-z]*"],
    "ALRS": [r"(?:^|[^а-яa-zё])алрос[а-яё]*", r"(?:^|[^а-яa-zё])alrosa[a-z]*"],
    "CHMF": [r"(?:^|[^а-яa-zё])северстал[а-яё]*", r"(?:^|[^а-яa-zё])severstal[a-z]*", r"(?:^|[^а-яa-zё])chmf"],
    "GAZP": [r"(?:^|[^а-яa-zё])газпром[а-яё]*", r"(?:^|[^а-яa-zё])gazprom[a-z]*"],
    "GMKN": [
        r"(?:^|[^а-яa-zё])норникел[а-яё]*",
        r"(?:^|[^а-яa-zё])норильск[а-яё]*\sникел[а-яё]*",
        r"(?:^|[^а-яa-zё])nornickel[a-z]*",
        r"(?:^|[^а-яа-zё])gmkn"
    ],
    "LKOH": [r"(?:^|[^а-яa-zё])лукойл[а-яё]*", r"(?:^|[^а-яa-zё])lukoil[a-z]*"],
    "MAGN": [
        r"(?:^|[^а-яa-zё])ммк",
        r"(?:^|[^а-яa-zё])магнитогорск[а-яё]*\sметаллургическ[а-яё]*",
        r"(?:^|[^а-яa-zё])magn[a-z]*"
    ],
    "MGNT": [r"(?:^|[^а-яa-zё])магнит[а-яё]*", r"(?:^|[^а-яa-zё])magnit[a-z]*"],
    "MOEX": [r"(?:^|[^а-яa-zё])московск[а-яё]*\sбирж[а-яё]*", r"(?:^|[^а-яa-zё])moex[a-z]*"],
    "MTSS": [r"(?:^|[^а-яa-zё])мтс", r"(?:^|[^а-яa-zё])mts", r"(?:^|[^а-яa-zё])mobile\s+telesystems"],
    "NVTK": [r"(?:^|[^а-яa-zё])новат[еэ]к[а-яё]*", r"(?:^|[^а-яa-zё])novatek[a-z]*"],
    "PHOR": [r"(?:^|[^а-яa-zё])фосагр[а-яё]*", r"(?:^|[^а-яa-zё])phosagro[a-z]*"],
    "PLZL": [
        r"(?:^|[^а-яa-zё])полюс[а-яё]*\sзолот[а-яё]*",
        r"(?:^|[^а-яa-zё])polyus[a-z]*",
        r"(?:^|[^а-яa-zё])plzl"
    ],
    "ROSN": [r"(?:^|[^а-яa-zё])роснефт[а-яё]*", r"(?:^|[^а-яa-zё])rosneft[a-z]*"],
    "RUAL": [r"(?:^|[^а-яa-zё])русал[а-яё]*", r"(?:^|[^а-яa-zё])rusal[a-z]*"],
    "SBER": [
        r"(?:^|[^а-яa-zё])сбербанк[а-яё]*",
        r"(?:^|[^а-яa-zё])сбер[а-яё]*",
        r"(?:^|[^а-яa-zё])sber[a-z]*",
        r"(?:^|[^а-яa-zё])sberbank[a-z]*"
    ],
    "SIBN": [
        r"(?:^|[^а-яa-zё])газпром\s+нефт[а-яё]*",
        r"(?:^|[^а-яa-zё])gazprom\s+neft[a-z]*"
    ],
    "T": [
        r"(?:^|[^а-яa-zё])т-?банк[а-яё]*",
        r"(?:^|[^а-яa-zё])тинькофф[а-яё]*",
        r"(?:^|[^а-яa-zё])tinkoff[a-z]*",
        r"(?:^|[^а-яa-zё])t-?bank[a-z]*"
    ],
    "VTBR": [r"(?:^|[^а-яa-zё])втб", r"(?:^|[^а-яa-zё])vtb[a-z]*"],
}

def extract_tickers(text, mapping=ticker_patterns):
    text = str(text).lower()
    found = []
    for ticker, patterns in mapping.items():
        if any(re.search(p, text) for p in patterns):
            found.append(ticker)
    return list(set(found)) if found else None

train_news_path = "news.csv"
test_news_path = "news_2.csv"

train_news = pd.read_csv(train_news_path)
test_news = pd.read_csv(test_news_path)

def add_tickers_column(df):
    if "title" in df.columns and "publication" in df.columns:
        text_data = df["title"].fillna("") + " " + df["publication"].fillna("")
    elif "title" in df.columns:
        text_data = df["title"].fillna("")
    else:
        text_data = df.astype(str).agg(" ".join, axis=1)
    df["tickers"] = text_data.apply(extract_tickers)
    return df

train_news = add_tickers_column(train_news)
test_news = add_tickers_column(test_news)

out_dir = Path("data/processed/")
out_dir.mkdir(parents=True, exist_ok=True)

train_out = "train_news_with_tickers.csv"
test_out = "test_news_with_tickers.csv"

train_news.to_csv(train_out, index=False)
test_news.to_csv(test_out, index=False)

print("✅ Файлы успешно сохранены:")

import os
import math
from dataclasses import dataclass, replace
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

__all__ = [
    "FinBertConfig",
    "RUN_FLAGS",
    "process_news_with_config",
    "run_with_flags",
    "detect_device",
]

@dataclass
class FinBertConfig:
    input_csv: str
    output_csv: str
    no_translate: bool = False
    batch_size: int = 64
    max_seq_len: int = 256
    max_new_tokens: int = 96
    truncate_publication: Optional[int] = 300
    use_ctranslate2: bool = False
    ctranslate2_dir: Optional[str] = None
    title_only: bool = False

RUN_FLAGS = FinBertConfig(
    input_csv="train_news_with_tickers.csv",
    output_csv="fast_train_news_with_sent.csv",
    no_translate=False,
    batch_size=64,
    max_seq_len=256,
    max_new_tokens=96,
    truncate_publication=300,
    use_ctranslate2=False,
    ctranslate2_dir=None,
    title_only=False,
)

RUN_FLAGS_TEST = FinBertConfig(
    input_csv="test_news_with_tickers.csv",
    output_csv="fast_test_news_with_sent.csv",
    no_translate=False,
    batch_size=64,
    max_seq_len=256,
    max_new_tokens=96,
    truncate_publication=300,
    use_ctranslate2=False,
    ctranslate2_dir=None,
    title_only=False,
)

def detect_device():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] torch={torch.__version__} | device={dev} | cuda={getattr(torch.version, 'cuda', None)}")
    if dev == "cuda":
        try:
            print(f"[GPU] {torch.cuda.get_device_name(0)} | mem={(torch.cuda.get_device_properties(0).total_memory/1e9):.1f} GB")
        except Exception as e:
            print(f"[GPU warn] {e}")
    return dev

def build_text(title, body, truncate_publication: Optional[int] = None):
    t = (str(title) if pd.notna(title) else "").strip()
    b = (str(body) if pd.notna(body) else "").strip()
    if truncate_publication and isinstance(truncate_publication, int) and truncate_publication > 0:
        b = b[:truncate_publication]
    if t and b:
        return f"{t}. {b}"
    return t or b

def safe_read_csv(path, encodings=("utf-8", "utf-8-sig", "cp1251")) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Не удалось прочитать CSV '{path}'. Последняя ошибка: {last_err}")

def load_translation_model(device: str, model_name="Helsinki-NLP/opus-mt-ru-en"):
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(device)
    if device == "cuda":
        mdl.half()
    else:
        from torch.quantization import quantize_dynamic
        mdl = quantize_dynamic(mdl, {nn.Linear}, dtype=torch.qint8)
    mdl.eval()
    print("[OK] Translation model loaded:", model_name)
    print("Translator on:", next(mdl.parameters()).device)
    return tok, mdl

def translate_texts(texts: List[str], tok, mdl, device: str,
                    batch_size=32, max_len=512, max_new_tokens=128) -> List[str]:
    res = []
    total_batches = math.ceil(len(texts) / batch_size)
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Translate RU→EN (HF greedy)"):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", truncation=True, padding=True,
                     max_length=max_len, pad_to_multiple_of=8).to(device)
        with torch.inference_mode():
            gen = mdl.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )
        out = tok.batch_decode(gen, skip_special_tokens=True)
        res.extend(out)
    return res

def translate_ct2(texts: List[str], model_dir: str, device: str,
                  batch_size=128, max_len=256, max_new_tokens=96) -> List[str]:
    try:
        import ctranslate2
        import sentencepiece as spm
    except Exception as e:
        raise RuntimeError("Для use_ctranslate2 нужны пакеты: ctranslate2 и sentencepiece") from e
    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))
    sp_tgt.load(os.path.join(model_dir, "target.spm"))
    compute_type = "float16" if device == "cuda" else "int8"
    translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)
    res = []
    total_batches = math.ceil(len(texts) / batch_size)
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=f"Translate RU→EN (ct2 {compute_type})"):
        batch = texts[i:i+batch_size]
        src_tokens = [sp_src.encode(t, out_type=str)[:max_len] for t in batch]
        out = translator.translate_batch(src_tokens, beam_size=1, max_decoding_length=max_new_tokens)
        res.extend(sp_tgt.decode_pieces(o.hypotheses[0]) for o in out)
    return res

def load_finbert(device: str, model_name="ProsusAI/finbert"):
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).to(device)
    if device == "cuda":
        mdl.half()
    else:
        from torch.quantization import quantize_dynamic
        mdl = quantize_dynamic(mdl, {nn.Linear}, dtype=torch.qint8)
    mdl.eval()
    try:
        from optimum.bettertransformer import BetterTransformer
        mdl = BetterTransformer.transform(mdl, keep_original_model=False)
        print("[BT] BetterTransformer enabled")
    except Exception:
        pass
    try:
        if device == "cuda":
            mdl = torch.compile(mdl, mode="max-autotune")
            print("[Compile] torch.compile enabled")
    except Exception:
        pass
    id2label = {i: mdl.config.id2label[i] for i in range(mdl.config.num_labels)}
    label_names = [id2label[i].lower() for i in range(mdl.config.num_labels)]
    print("[OK] FinBERT loaded:", model_name, "| labels:", label_names)
    print("FinBERT on:", next(mdl.parameters()).device)
    return tok, mdl, label_names

def load_multilingual_sentiment(device: str, model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).to(device)
    if device == "cuda":
        mdl.half()
    else:
        from torch.quantization import quantize_dynamic
        mdl = quantize_dynamic(mdl, {nn.Linear}, dtype=torch.qint8)
    mdl.eval()
    try:
        from optimum.bettertransformer import BetterTransformer
        mdl = BetterTransformer.transform(mdl, keep_original_model=False)
        print("[BT] BetterTransformer enabled (multi)")
    except Exception:
        pass
    try:
        if device == "cuda":
            mdl = torch.compile(mdl, mode="max-autotune")
            print("[Compile] torch.compile enabled (multi)")
    except Exception:
        pass
    id2label = {i: mdl.config.id2label[i] for i in range(mdl.config.num_labels)}
    label_names = [id2label[i].lower() for i in range(mdl.config.num_labels)]
    print("[OK] Multilingual sentiment loaded:", model_name, "| labels:", label_names)
    print("Sentiment model on:", next(mdl.parameters()).device)
    return tok, mdl, label_names

def run_text_classifier(texts: List[str], tok, mdl, device: str,
                        batch_size=64, max_len=512, desc="Classifier") -> np.ndarray:
    total_batches = math.ceil(len(texts) / batch_size)
    out_chunks = []
    for start in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=desc):
        batch = texts[start:start+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                  max_length=max_len, pad_to_multiple_of=8).to(device)
        with torch.inference_mode():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        out_chunks.append(probs)
    return np.vstack(out_chunks) if out_chunks else np.empty((0, 0))

def process_news_with_config(cfg: FinBertConfig) -> pd.DataFrame:
    device = detect_device()
    df = safe_read_csv(cfg.input_csv)
    required_cols = {"publish_date", "title", "publication", "tickers"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {missing}")
    if cfg.title_only:
        df["__text_ru"] = df["title"].fillna("").astype(str)
    else:
        df["__text_ru"] = [
            build_text(t, p, truncate_publication=cfg.truncate_publication)
            for t, p in zip(df["title"], df["publication"])
        ]
    text_codes, unique_texts = pd.factorize(df["__text_ru"], sort=False)
    unique_texts = list(unique_texts)
    if not cfg.no_translate:
        if cfg.use_ctranslate2:
            if not cfg.ctranslate2_dir or not os.path.isdir(cfg.ctranslate2_dir):
                raise ValueError("use_ctranslate2=True, но ctranslate2_dir не задан или не существует")
            unique_texts_en = translate_ct2(
                unique_texts, cfg.ctranslate2_dir, device,
                batch_size=max(64, cfg.batch_size),
                max_len=cfg.max_seq_len,
                max_new_tokens=cfg.max_new_tokens,
            )
        else:
            trans_tok, trans_mdl = load_translation_model(device)
            unique_texts_en = translate_texts(
                unique_texts, trans_tok, trans_mdl, device,
                batch_size=cfg.batch_size,
                max_len=cfg.max_seq_len,
                max_new_tokens=cfg.max_new_tokens,
            )
        texts_en = [unique_texts_en[c] for c in text_codes]
    else:
        texts_en = None
    if not cfg.no_translate:
        fb_tok, fb_mdl, label_names = load_finbert(device)
        probs_unique = run_text_classifier(
            unique_texts_en, fb_tok, fb_mdl, device,
            batch_size=cfg.batch_size, max_len=cfg.max_seq_len, desc="FinBERT (EN)"
        )
        probs_out = probs_unique[text_codes]
    else:
        ml_tok, ml_mdl, label_names = load_multilingual_sentiment(device)
        probs_unique = run_text_classifier(
            unique_texts, ml_tok, ml_mdl, device,
            batch_size=cfg.batch_size, max_len=cfg.max_seq_len, desc="Multilingual sentiment (RU)"
        )
        probs_out = probs_unique[text_codes]
    out_df = df.copy()
    for j, name in enumerate(label_names):
        out_df[f"p_{name.lower()}"] = probs_out[:, j]
    ln = [n.lower() for n in label_names]
    idx_pos = ln.index("positive") if "positive" in ln else None
    idx_neg = ln.index("negative") if "negative" in ln else None
    out_df["sent_label"] = [label_names[int(i)] for i in np.argmax(probs_out, axis=1)]
    if idx_pos is not None and idx_neg is not None:
        out_df["sent_score"] = probs_out[:, idx_pos] - probs_out[:, idx_neg]
    else:
        out_df["sent_score"] = np.nan
    out_df.drop(columns=["__text_ru"], inplace=True, errors="ignore")
    out_df.to_csv(cfg.output_csv + ".partial.csv", index=False)
    out_df.to_csv(cfg.output_csv, index=False)
    print(f"[OK] Saved: {cfg.output_csv}")
    print(f"[Note] Partial file: {cfg.output_csv}.partial.csv (оставлен)")
    return out_df

def run_with_flags(**overrides) -> pd.DataFrame:
    cfg = replace(RUN_FLAGS, **overrides)
    return process_news_with_config(cfg)

print("[RUN] finbert.py запускается с параметрами RUN_FLAGS:")
print(RUN_FLAGS)
process_news_with_config(RUN_FLAGS)
process_news_with_config(RUN_FLAGS_TEST)

def _pick_device(device: str = "auto"):
    import importlib
    import numpy as _np
    import pandas as _pd
    from scipy import sparse as _sp
    dev = "cpu"
    mods = {}
    if device in ("auto", "gpu"):
        try:
            _cp = importlib.import_module("cupy")
            _cudf = importlib.import_module("cudf")
            _cuml = importlib.import_module("cuml")
            _cupyx_sp = importlib.import_module("cupyx.scipy.sparse")
            tfidf_cls = importlib.import_module("cuml.feature_extraction.text").TfidfVectorizer
            svd_cls = importlib.import_module("cuml.decomposition").TruncatedSVD
            from sklearn.decomposition import NMF as _skNMF
            def _to_cpu_sparse(x):
                return x.get()
            def _to_cpu(x):
                return _cp.asnumpy(x)
            def _to_cudf_series(ps):
                return _cudf.Series(ps.astype(str).fillna(""))
            dev = "gpu"
            mods.update(dict(
                xp=_cp, pd=_pd, sp=_cupyx_sp,
                tfidf_cls=tfidf_cls, svd_cls=svd_cls, nmf_cls=_skNMF,
                to_cpu_sparse=_to_cpu_sparse, to_cpu=_to_cpu, to_cudf_series=_to_cudf_series,
                cudf=_cudf, cuml=_cuml
            ))
        except Exception:
            pass
    if dev == "cpu":
        from sklearn.feature_extraction.text import TfidfVectorizer as _skTFIDF
        from sklearn.decomposition import TruncatedSVD as _skSVD, NMF as _skNMF
        def _to_cpu_sparse(x): return x
        def _to_cpu(x): return x
        def _to_cudf_series(ps): return ps
        mods.update(dict(
            xp=_np, pd=_pd, sp=_sp,
            tfidf_cls=_skTFIDF, svd_cls=_skSVD, nmf_cls=_skNMF,
            to_cpu_sparse=_to_cpu_sparse, to_cpu=_to_cpu, to_cudf_series=_to_cudf_series
        ))
    return dev, mods

def hash_embed_publication(pub: str, dim: int = 8, seed: int = 42):
    rnd = np.random.RandomState(abs(hash((pub, seed))) % (2**32))
    v = rnd.normal(size=dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v

def softmax(x, xp=None):
    if xp is None:
        import numpy as np
        xp = np
    x = xp.asarray(x, dtype=xp.float32)
    if x.size:
        x = x - x.max()
    ex = xp.exp(x)
    return ex / (ex.sum() + 1e-12)

def transformer_like_pool(embs, times, now_ts, tau_days: float = 10.0, time_bias: float = 1.0, xp=None):
    import pandas as pd
    if xp is None:
        import numpy as _np
        xp = _np
    embs = xp.asarray(embs, dtype=xp.float32)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    if embs.shape[0] == 0:
        return xp.zeros((embs.shape[1],), dtype=xp.float32)
    D = embs.shape[1]
    q = embs.mean(axis=0).astype(xp.float32)
    scores = (embs @ q) / (xp.sqrt(D) + 1e-9)
    times_ns = pd.to_datetime(times).values.astype('datetime64[ns]')
    now_ns   = pd.Timestamp(now_ts).to_datetime64().astype('datetime64[ns]')
    dt_days_cpu = ((now_ns - times_ns).astype('timedelta64[s]').astype('float64') / 86400.0)
    dt_days_cpu = dt_days_cpu.clip(min=0.0)
    dt_days = xp.asarray(dt_days_cpu, dtype=xp.float32)
    weights = softmax(scores + time_bias * (-dt_days / max(1e-6, tau_days)), xp=xp)
    ctx = (weights[:, None] * embs).sum(axis=0).astype(xp.float32)
    return ctx

def _norm_ticker(x: str) -> str:
    return re.sub(r"\s+", "", str(x).upper())

def _parse_tickers_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, set)):
        seq = list(x)
    else:
        s = str(x).strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                seq = list(v)
            else:
                seq = [s]
        except Exception:
            seq = re.split(r"[,\|\s;]+", s.strip("[](){}'\" "))
    return [_norm_ticker(t) for t in seq if str(t).strip()]

class NewsVectorizer:
    def __init__(self, text_dim=128, n_topics=16, max_features=30000, random_state=42, device: str = "auto"):
        self.text_dim = text_dim
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self.device, self.mods = _pick_device(device)
        self.vec = None
        self.svd = None
        self.nmf = None

    def fit(self, news_texts: pd.Series):
        pd = self.mods["pd"]
        tfidf_cls = self.mods["tfidf_cls"]
        nmf_cls = self.mods["nmf_cls"]
        to_cpu_sparse = self.mods["to_cpu_sparse"]
        to_cudf_series = self.mods["to_cudf_series"]
        from sklearn.decomposition import TruncatedSVD as _skSVD
        if self.device == "gpu":
            texts_gpu = to_cudf_series(news_texts.astype(str).fillna(''))
            self.vec = tfidf_cls(max_features=self.max_features, ngram_range=(1,2), min_df=2)
            X_gpu = self.vec.fit_transform(texts_gpu)
            X_cpu = to_cpu_sparse(X_gpu)
            self.svd = _skSVD(n_components=self.text_dim, random_state=self.random_state)
            self.svd.fit(X_cpu)
            if self.n_topics and self.n_topics > 0:
                self.nmf = nmf_cls(n_components=self.n_topics, init='nndsvda',
                                random_state=self.random_state, max_iter=300)
                self.nmf.fit(X_cpu)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer as _skTFIDF
            from sklearn.decomposition import TruncatedSVD as _skSVD
            self.vec = _skTFIDF(max_features=self.max_features, ngram_range=(1,2), min_df=2)
            X = self.vec.fit_transform(news_texts.astype(str).fillna(''))
            self.svd = _skSVD(n_components=self.text_dim, random_state=self.random_state)
            self.svd.fit(X)
            if self.n_topics and self.n_topics > 0:
                self.nmf = nmf_cls(n_components=self.n_topics, init='nndsvda',
                                random_state=self.random_state, max_iter=300)
                self.nmf.fit(X)
        return self

    def transform(self, texts: pd.Series):
        import numpy as np
        to_cpu_sparse = self.mods["to_cpu_sparse"]
        to_cudf_series = self.mods["to_cudf_series"]
        if self.device == "gpu":
            texts_gpu = to_cudf_series(texts.astype(str).fillna(''))
            X_gpu = self.vec.transform(texts_gpu)
            X_cpu = to_cpu_sparse(X_gpu)
            xs = self.svd.transform(X_cpu).astype(np.float32)
            if self.nmf is not None:
                xt = self.nmf.transform(X_cpu).astype(np.float32)
                xt /= (xt.sum(axis=1, keepdims=True) + 1e-9)
            else:
                xt = np.zeros((xs.shape[0], self.n_topics), dtype=np.float32)
            return xs, xt
        else:
            X = self.vec.transform(texts.astype(str).fillna(''))
            xs = self.svd.transform(X).astype(np.float32)
            if self.nmf is not None:
                xt = self.nmf.transform(X).astype(np.float32)
                xt /= (xt.sum(axis=1, keepdims=True) + 1e-9)
            else:
                xt = np.zeros((xs.shape[0], self.n_topics), dtype=np.float32)
            return xs, xt

def make_extra_news_features(day_df: pd.DataFrame, last_60d_df: pd.DataFrame | None, n_topics: int):
    feats = {}
    cnt = len(day_df)
    feats['news_count_day'] = int(cnt)
    if cnt == 0:
        feats['dup_ratio'] = 0.0
        feats['title_len_mean'] = 0.0
        feats['title_len_std']  = 0.0
        feats['publication_len_mean'] = 0.0
        feats['publication_len_std']  = 0.0
        feats['top_topic_idx'] = -1
        feats['top_topic_strength'] = 0.0
        for i in range(n_topics):
            feats[f'topic_mean_{i}'] = 0.0
        feats['news_spike_rough'] = 0.0
        return feats
    txt = (day_df['title'].astype(str).str.lower().str.strip() + ' ' +
           day_df['publication'].astype(str).str.lower().str.strip()).str.replace(r'\s+', ' ', regex=True)
    uniq = int(txt.nunique())
    feats['dup_ratio'] = float(1.0 - uniq / max(1, cnt))
    tlen = day_df['title'].astype(str).str.len()
    plen = day_df['publication'].astype(str).str.len()
    feats['title_len_mean'] = float(tlen.mean())
    feats['title_len_std']  = float(tlen.std() or 0.0)
    feats['publication_len_mean'] = float(plen.mean())
    feats['publication_len_std']  = float(plen.std() or 0.0)
    topic_mat = np.stack(day_df['topic_vec'].values, axis=0)
    topic_mean = topic_mat.mean(axis=0)
    top_idx = int(np.argmax(topic_mean))
    feats['top_topic_idx'] = top_idx
    feats['top_topic_strength'] = float(topic_mean[top_idx])
    for i, v in enumerate(topic_mean):
        feats[f'topic_mean_{i}'] = float(v)
    if last_60d_df is not None and len(last_60d_df):
        days = last_60d_df['publish_date'].dt.normalize().nunique()
        avg_per_day = float(len(last_60d_df)) / max(1, int(days))
        feats['news_spike_rough'] = float(cnt / (avg_per_day + 1e-9))
    else:
        feats['news_spike_rough'] = float(cnt)
    return feats

def add_news_context_simple(
    candles_csv: str,
    news_csv: str,
    out_csv: str = None,
    text_dim: int = 128,
    n_topics: int = 16,
    max_news_per_sample: int = 128,
    tau_days: float = 10.0,
    add_aditional_emb: bool = True,
    add_context_emb: bool = True,
    device: str = "auto"
):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    dev, mods = _pick_device(device)
    xp = mods["xp"]
    candles = pd.read_csv(candles_csv)
    candles['begin'] = pd.to_datetime(candles['begin'])
    candles['date'] = candles['begin'].dt.normalize()
    candles['ticker_norm'] = candles['ticker'].apply(_norm_ticker)
    news = pd.read_csv(news_csv)
    news['publish_date'] = pd.to_datetime(news['publish_date'])
    news['tickers_list'] = news['tickers'].apply(_parse_tickers_cell)
    out_parts = [candles]
    if add_context_emb:
        vec_title = NewsVectorizer(text_dim=text_dim, n_topics=n_topics, device=device).fit(
            news['title'].astype(str).fillna('')
        )
        vec_pub   = NewsVectorizer(text_dim=text_dim, n_topics=n_topics, device=device).fit(
            news['publication'].astype(str).fillna('')
        )
        xs_t, xt_t = vec_title.transform(news['title'].astype(str).fillna(''))
        xs_p, xt_p = vec_pub.transform(news['publication'].astype(str).fillna(''))
        news_ctx = news[['publish_date','tickers_list']].copy()
        emb_np = np.concatenate([xs_t, xt_t, xs_p, xt_p], axis=1).astype(np.float32)
        news_ctx['emb'] = list(emb_np)
        D = 2*text_dim + 2*n_topics
        ctx_cols = [f'ctx_{i}' for i in range(D)]
        ctx_rows, diag_counts = [], []
        for row in candles.itertuples(index=False):
            tkr = row.ticker_norm
            cutoff = row.date - pd.Timedelta(days=1)
            mask = (news_ctx['publish_date'] <= cutoff) & (news_ctx['tickers_list'].apply(lambda lst: tkr in lst))
            past = news_ctx[mask]
            diag_counts.append(int(len(past)))
            if len(past) > max_news_per_sample:
                past = past.tail(max_news_per_sample)
            if len(past):
                E = np.stack(past['emb'].values, axis=0).astype(np.float32)
                if dev == "gpu":
                    import cupy as cp
                    E_xp = cp.asarray(E)
                    T = pd.to_datetime(past['publish_date']).values.astype('datetime64[ns]')
                    ctx_xp = transformer_like_pool(E_xp, T, cutoff, tau_days=tau_days, time_bias=1.0, xp=cp)
                    ctx = cp.asnumpy(ctx_xp)
                else:
                    T = pd.to_datetime(past['publish_date']).values.astype('datetime64[ns]')
                    ctx = transformer_like_pool(E, T, cutoff, tau_days=tau_days, time_bias=1.0)
            else:
                ctx = np.zeros((D,), dtype=np.float32)
            ctx_rows.append(ctx)
        ctx_df = pd.DataFrame(np.vstack(ctx_rows), columns=ctx_cols, index=candles.index)
        out_parts += [ctx_df, pd.DataFrame({'ctx_news_count': diag_counts}, index=candles.index)]
    if add_aditional_emb:
        vec_feat_t = NewsVectorizer(text_dim=text_dim, n_topics=n_topics, device=device).fit(
            news['title'].astype(str).fillna('')
        )
        _, xt_t = vec_feat_t.transform(news['title'].astype(str).fillna(''))
        vec_feat_p = NewsVectorizer(text_dim=text_dim, n_topics=n_topics, device=device).fit(
            news['publication'].astype(str).fillna('')
        )
        _, xt_p = vec_feat_p.transform(news['publication'].astype(str).fillna(''))
        topic_vec = (xt_t + xt_p) / 2.0
        news_feat = news[['publish_date','title','publication','tickers_list']].copy()
        news_feat['topic_vec'] = list(topic_vec.astype(np.float32))
        extra_cols = (
            ['news_count_day','dup_ratio',
             'title_len_mean','title_len_std',
             'publication_len_mean','publication_len_std',
             'top_topic_idx','top_topic_strength','news_spike_rough'] +
            [f'topic_mean_{i}' for i in range(n_topics)]
        )
        extra_rows = []
        for row in candles.itertuples(index=False):
            tkr = row.ticker_norm
            cutoff = row.date - pd.Timedelta(days=1)
            mask_has = news_feat['tickers_list'].apply(lambda lst: tkr in lst)
            day_news = news_feat[(news_feat['publish_date'].dt.normalize() == cutoff) & mask_has].copy()
            last60   = news_feat[(news_feat['publish_date'] > (cutoff - pd.Timedelta(days=60))) &
                                 (news_feat['publish_date'] <= cutoff) & mask_has].copy()
            extra = make_extra_news_features(day_news, last60, n_topics)
            extra_rows.append([extra.get(c, 0.0) for c in extra_cols])
        extra_df = pd.DataFrame(extra_rows, columns=extra_cols, index=candles.index)
        out_parts.append(extra_df)
    enriched = pd.concat(out_parts, axis=1).drop(columns=['ticker_norm'], errors='ignore')
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(out_csv, index=False)
    return enriched

enriched = add_news_context_simple(
    candles_csv="candles.csv",
    news_csv="fast_train_news_with_sent.csv",
    out_csv="candles_features.csv",
    text_dim=64,
    n_topics=12,
    max_news_per_sample=128,
    tau_days=7.0,
    add_context_emb = False,
    device="auto",
)

enriched = add_news_context_simple(
    candles_csv="candles_2.csv",
    news_csv="fast_test_news_with_sent.csv",
    out_csv="candles_2_features.csv",
    text_dim=64,
    n_topics=12,
    max_news_per_sample=128,
    tau_days=7.0,
    add_context_emb = False,
    device="auto",
)

def _safe_parse_tickers(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [str(x).strip().upper() for x in obj if str(x).strip()]
    except Exception:
        pass
    s = re.sub(r"[\[\]'\"{}]", " ", s)
    tokens = re.split(r"[^A-Za-z0-9\.]+", s)
    return [t.upper() for t in tokens if t.strip()]

def make_targets(df: pd.DataFrame, max_horizon: int = 20) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["ticker", "begin"]).reset_index(drop=True)
    g = out.groupby("ticker", group_keys=False)
    for i in range(1, max_horizon + 1):
        out[f"R_{i}"] = g["close"].shift(-i) / out["close"] - 1.0
    return out

def build_news_features(
    news_csv: str | pd.DataFrame,
    roll_days=(3, 7),
    use_cols=("p_positive", "p_negative", "p_neutral", "sent_score"),
):
    print("📥 Loading news...")
    if isinstance(news_csv, (str, Path)):
        news = pd.read_csv(news_csv)
    else:
        news = news_csv.copy()
    news = news.drop(columns=["Unnamed: 0"], errors="ignore")
    news["publish_date"] = pd.to_datetime(news["publish_date"], errors="coerce")
    news = news.dropna(subset=["publish_date"])
    news["date"] = news["publish_date"].dt.normalize()
    tqdm.pandas(desc="🧷 Parse tickers")
    news["ticker_list"] = news["tickers"].progress_apply(_safe_parse_tickers)
    news = news.explode("ticker_list").rename(columns={"ticker_list": "ticker"})
    news["ticker"] = news["ticker"].astype(str)
    news = news[news["ticker"].str.len() > 0]
    keep = ["ticker", "date", *use_cols]
    news = news[keep].copy()
    agg_map = {
        "sent_score": ["mean", "max", "min", "sum"],
        "p_positive": ["mean"],
        "p_negative": ["mean"],
        "p_neutral": ["mean"],
    }
    print("📊 Daily aggregations...")
    daily = (
        news.groupby(["ticker", "date"])
        .agg(agg_map)
        .rename(columns={"mean": "mean", "max": "max", "min": "min", "sum": "sum"})
    )
    daily.columns = [f"{c0}_{c1}" for (c0, c1) in daily.columns.to_flat_index()]
    daily = daily.reset_index()
    cnt = news.groupby(["ticker", "date"]).size().reset_index(name="n_news")
    daily = daily.merge(cnt, on=["ticker", "date"], how="left")
    for c in ["sent_score_mean", "sent_score_max", "sent_score_min", "sent_score_sum",
              "p_positive_mean", "p_negative_mean", "p_neutral_mean", "n_news"]:
        if c not in daily.columns:
            daily[c] = 0.0
    if roll_days:
        print("🔁 Rolling features...")
        pieces = []
        g = daily.sort_values(["ticker", "date"]).groupby("ticker", group_keys=False)
        for ticker, df_t in tqdm(g, total=g.ngroups, desc="Rolling per ticker"):
            df_t = df_t.sort_values("date").copy()
            base_cols = [
                "sent_score_mean", "sent_score_max", "sent_score_min", "sent_score_sum",
                "p_positive_mean", "p_negative_mean", "p_neutral_mean", "n_news",
            ]
            for col in base_cols:
                df_t[f"{col}_lag1"] = df_t[col].shift(1)
            for R in roll_days:
                for col in base_cols:
                    df_t[f"{col}_roll{R}"] = (
                        df_t[col].rolling(window=R, min_periods=1).mean().shift(1)
                    )
            pieces.append(df_t)
        daily = pd.concat(pieces, ignore_index=True)
    return daily

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ds"] = pd.to_datetime(out["date"])
    out["dow"] = out["ds"].dt.weekday
    out["month"] = out["ds"].dt.month
    out["is_month_start"] = out["ds"].dt.is_month_start.astype(int)
    out["is_month_end"]   = out["ds"].dt.is_month_end.astype(int)
    out["dow_sin"] = np.sin(2*np.pi*out["dow"]/7.0)
    out["dow_cos"] = np.cos(2*np.pi*out["dow"]/7.0)
    return out

def build_xy(df_feat: pd.DataFrame, max_horizon: int = 20, drop_today_news: bool = True):
    targets = [f"R_{i}" for i in range(1, max_horizon + 1)]
    df = df_feat.dropna(subset=targets).copy()
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.drop(columns=[c for c in X.columns if c.startswith("R_")], errors="ignore")
    X = X.drop(columns=["Unnamed: 0"], errors="ignore")
    if drop_today_news:
        base_news_today = [
            "sent_score_mean","sent_score_max","sent_score_min","sent_score_sum",
            "p_positive_mean","p_negative_mean","p_neutral_mean","n_news",
        ]
        to_drop = [c for c in X.columns if c in base_news_today]
        X = X.drop(columns=to_drop, errors="ignore")
    Y = df[targets].astype(float).copy()
    return df, X, Y

def get_pred(X_train, y_train, X_test, model_kind="rf", random_state=42):
    if model_kind == "gbr":
        base = GradientBoostingRegressor(random_state=random_state)
    elif model_kind == "rf":
        base = RandomForestRegressor(
            n_estimators=500, max_depth=None, n_jobs=-1, random_state=random_state
        )
    elif model_kind == "ridge":
        base = Ridge(alpha=1.0, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model.predict(X_train), model.predict(X_test)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, tag: str = "TEST"):
    H = y_true.shape[1]
    maes, accs = [], []
    for i in range(H):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        acc = (np.sign(y_true[:, i]) == np.sign(y_pred[:, i])).mean()
        maes.append(mae); accs.append(acc)
    print(f"\n=== METRICS ({tag}) ===")
    print("Horizon i :  MAE      |  Sign Acc")
    for i, (m, a) in enumerate(zip(maes, accs), start=1):
        print(f"R_{i:>2}:     {m:8.6f} |   {a:6.3f}")
    print(f"AVG:       {np.mean(maes):8.6f} |   {np.mean(accs):6.3f}")
    return np.mean(maes), np.mean(accs)

def generate_submissions(
    candles_path: str = "candles.csv",
    candles2_path: str = "candles_2.csv",
    news_path: str | None = "fast_train_news_with_sent.csv",
    news2_path: str | None = "fast_test_news_with_sent.csv",
    max_horizon: int = 20,
    submission_csv: str = "submissions.csv",
    news_roll_days=(3,7),
    drop_today_news: bool = True,
    model_kind: str = "rf"
):
    df_tr = pd.read_csv(candles_path)
    df_te = pd.read_csv(candles2_path)
    for d in (df_tr, df_te):
        d["begin"] = pd.to_datetime(d["begin"], errors="coerce")
        d["ticker"] = d["ticker"].astype(str).str.upper()
        d["date"] = d["begin"].dt.normalize()
    news_feats_tr = build_news_features(news_path) if news_path is not None else pd.DataFrame(columns=["ticker","date"])
    news_feats_te = build_news_features(news2_path) if news2_path is not None else pd.DataFrame(columns=["ticker","date"])
    df_tr_lbl = make_targets(df_tr, max_horizon=max_horizon)
    df_trm = df_tr_lbl.merge(news_feats_tr, on=["ticker","date"], how="left")
    df_trm = add_calendar_features(df_trm)
    df_train, X_train, Y_train = build_xy(df_trm, max_horizon=max_horizon, drop_today_news=drop_today_news)
    fillna_vals = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fillna_vals)
    te_sorted = df_te.sort_values(["ticker", "begin"])
    te_last = te_sorted.groupby("ticker").tail(1).copy()
    df_tem = te_last.merge(news_feats_te, on=["ticker","date"], how="left")
    df_tem = add_calendar_features(df_tem)
    X_test = df_tem.select_dtypes(include=[np.number]).copy()
    X_test = X_test.drop(columns=[c for c in X_test.columns if c.startswith("R_")], errors="ignore")
    X_test = X_test.drop(columns=["Unnamed: 0"], errors="ignore")
    if drop_today_news:
        base_news_today = [
            "sent_score_mean","sent_score_max","sent_score_min","sent_score_sum",
            "p_positive_mean","p_negative_mean","p_neutral_mean","n_news",
        ]
        X_test = X_test.drop(columns=[c for c in X_test.columns if c in base_news_today], errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.fillna(fillna_vals)
    print(f"👟 Train samples: {len(X_train)} | Test samples: {len(X_test)} | Features: {X_train.shape[1]}")
    pred_train, pred_test = get_pred(X_train, Y_train, X_test, model_kind=model_kind)
    evaluate(Y_train.values, pred_train, tag="TRAIN")
    sub = pd.DataFrame({"ticker": te_last["ticker"].values})
    for i in range(1, max_horizon + 1):
        sub[f"p{i}"] = pred_test[:, i - 1]
    sub = sub.sort_values("ticker").reset_index(drop=True)
    Path(submission_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_csv, index=False)
    print(f"✅ Saved {submission_csv} (rows: {len(sub)})")
    return sub

_ = generate_submissions(
    candles_path="candles_features.csv",
    candles2_path="candles_2_features.csv",
    news_path="fast_train_news_with_sent.csv",
    news2_path="fast_test_news_with_sent.csv",
    max_horizon=20,
    submission_csv="submissions.csv",
    news_roll_days=(3,7),
    drop_today_news=True,
    model_kind="rf",
)

