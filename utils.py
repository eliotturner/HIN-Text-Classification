#Utility methods which process the data into HIN, perform small functions, and display the graphs
#Developed by: Eliot Turner, Connor Balint, Yuth Yun, Anthony Marcic
#Description: Loads HIN of 20_newsgroups Dataset using "util" functions,
#             and trains a R-GCN using that HIN and tests the model.

#import necessary libraries
from __future__ import annotations
import glob, math, os, random, re
from collections import Counter
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import graphistry
from sentence_transformers import SentenceTransformer


#tokenizes words by spaces, excluding punctuation.
_pat = re.compile(r"\w+")
def tokenize(txt: str) -> list[str]:
    return _pat.findall(txt.lower())

#converts edge list into source and destination node tensors
def to_tensor(edges):
    if not edges:
        x = torch.empty(0, dtype=torch.int64)
        return x, x
    s, d = zip(*edges)
    return torch.tensor(s, dtype=torch.int64), torch.tensor(d, dtype=torch.int64)



#Splits email into headers and body
#example: "Subject: subj \n body" --> {'Subject' : 'subj', ...} & {body}
def _split_headers_body(msg: str):
    hdr, body, in_hdr = [], [], True
    for ln in msg.splitlines():
        if in_hdr and not ln.strip():
            in_hdr = False; continue
        (hdr if in_hdr else body).append(ln)

    hdrs, key = {}, None
    for ln in hdr:
        if ln.startswith((" ", "\t")) and key:
            hdrs[key] += " " + ln.strip()
        elif ":" in ln:
            k, v = ln.split(":", 1)
            hdrs[k.lower()] = v.strip(); key = k.lower()
    return hdrs, "\n".join(body).strip()


#read 20_newsgroups file system, and convert all/sample into dataframe format
def build_dataframe(root_dir: str, sample_size=2000, full_test=False):
    #recursively locate all docs
    all_files = [(fp, grp)
                 for grp in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, grp))
                 for fp in glob.glob(os.path.join(root_dir, grp, "*"))]
    #use all samples or random sample as params specify
    files = all_files if full_test else random.sample(all_files,
                                                      min(sample_size, len(all_files)))
    rows, dropped = [], 0
    for fp, grp in files:
        #read file, drop if not english
        try:
            raw = Path(fp).read_text(encoding="latin1")
        except Exception:
            dropped += 1; continue

        #split into headers and body
        hdrs, body = _split_headers_body(raw)

        #extract from, subject, and organization, drop if any are missing
        frm, subj, org = hdrs.get("from"), hdrs.get("subject"), hdrs.get("organization")
        if not (frm and subj and org and body):
            dropped += 1; continue

        #add row with that documents information
        rows.append({"sender": frm, "subject": subj, "organization": org,
                     "newsgroup": grp, "text": body.replace("\r", "")})
    return pd.DataFrame(rows), dropped, len(files)


#compute IDF for words, and trim words based on min_df and max_df_ratio to reduce redundancy (valid tokens)
def compute_idf(df: pd.DataFrame, min_df: int = 5, max_df_ratio: float = 0.5):
    N, df_cnt = len(df), Counter()
    for _, r in df.iterrows():
        toks = set(tokenize(r["text"])) | set(tokenize(r["subject"]))
        for t in toks:
            df_cnt[t] += 1
    valid = {t for t, c in df_cnt.items() if c >= min_df and c / N <= max_df_ratio}
    idf = {t: math.log((1 + N) / (1 + df_cnt[t])) + 1 for t in valid}
    return idf, valid


#build heterograph based on dataframe, idf, valid tokens.
#uses glove embedding
#graph mode indicates whether to include everything, everything but metadata, or only metadata.
def build_heterograph(
    df: pd.DataFrame,
    idf_dict: dict[str, float],
    valid_tokens: set[str],
    glove_path: str = "glove.6B.300d.txt",
    glove_dim: int = 300,
    seed: int = 42,
    graph_mode: str = "full",
):
    #Node flags based on mode
    graph_mode = graph_mode.lower()
    if graph_mode == "all":
        graph_mode = "full"
    assert graph_mode in {"full", "words", "metadata"}

    #control signals for inclusion
    include_metadata     = graph_mode in {"full", "metadata"}
    include_body_edges   = graph_mode in {"full", "words"}

    #random seed for learned embeddings, BERT for documents
    rng   = np.random.default_rng(seed)
    sbert = SentenceTransformer("all-MiniLM-L6-v2")  # 384‑D

    #ensures array has 512 cols
    def pad512(mat: np.ndarray) -> np.ndarray:
        return (np.hstack([mat, np.zeros((mat.shape[0], 512 - mat.shape[1]),
                                         dtype=np.float32)])
                if mat.shape[1] < 512 else mat[:, :512])

    #determine vocab
    if graph_mode == "metadata":
        # only tokens appearing in subject lines
        subject_tokens = set()
        for subj in df["subject"]:
            subject_tokens |= set(tokenize(subj))
        tokens_to_use = valid_tokens & subject_tokens
    else:
        tokens_to_use = valid_tokens

    words = sorted(tokens_to_use)
    w2i   = {w: i for i, w in enumerate(words)}
    V     = len(words)

    #apply GloVe embeddings to word nodes
    W = np.random.randn(V, glove_dim).astype(np.float32)
    if Path(glove_path).exists():
        with open(glove_path, "r", encoding="utf-8") as f:
            for ln in f:
                w, *vec = ln.split()
                if w in w2i:
                    W[w2i[w]] = np.asarray(vec, dtype=np.float32)
    #IDF scaling
    for w, idx in w2i.items():
        W[idx] *= idf_dict[w]

    #Metadata inclusion and BERT for subject
    if include_metadata:
        sender2i = {v: i for i, v in enumerate(df["sender"].unique())}
        org2i    = {v: i for i, v in enumerate(df["organization"].unique())}
        subjects = df["subject"].unique().tolist()
        subj2i   = {s: i for i, s in enumerate(subjects)}

        subj_raw = sbert.encode(subjects, convert_to_numpy=True,
                                  show_progress_bar=False)
        subj_vec = (subj_raw[:, :256] if subj_raw.shape[1] >= 256
                    else np.hstack([subj_raw,
                                    np.zeros((subj_raw.shape[0],
                                              256 - subj_raw.shape[1]),
                                             dtype=np.float32)]))
    else:
        sender2i = org2i = subj2i = {}
        subj_vec = np.empty((0, 256), dtype=np.float32)

    #document embeddings using bert
    if graph_mode == "metadata":
        doc_raw = sbert.encode(df["subject"].tolist(), batch_size=32,
                               convert_to_numpy=True, show_progress_bar=False)
    else:
        doc_raw = sbert.encode(df["text"].tolist(), batch_size=32,
                               convert_to_numpy=True, show_progress_bar=True)
    doc_vec = pad512(doc_raw)

    #edge holders
    doc_word_f, doc_word_r, doc_word_w = [], [], []
    subj_word_f, subj_word_r, subj_word_w = [], [], []
    doc_subj_f, doc_subj_r = [], []
    doc_sender_f, doc_sender_r = [], []
    sender_org_f, sender_org_r = [], []

    #build edges
    for doc_id, r in df.iterrows():
        if include_body_edges:
            tf_body = Counter(tokenize(r["text"]))
            for tok, cnt in tf_body.items():
                if tok in w2i:
                    idx = w2i[tok]
                    wt  = math.log1p(cnt)
                    doc_word_f.append((doc_id, idx))
                    doc_word_r.append((idx, doc_id))
                    doc_word_w.extend([wt, wt])

        if include_metadata:
            sid = subj2i[r["subject"]]
            doc_subj_f.append((doc_id, sid))
            doc_subj_r.append((sid, doc_id))

            send_idx = sender2i[r["sender"]]
            org_idx  = org2i[r["organization"]]
            doc_sender_f.append((doc_id, send_idx))
            doc_sender_r.append((send_idx, doc_id))
            sender_org_f.append((send_idx, org_idx))
            sender_org_r.append((org_idx, send_idx))

            tf_subj = Counter(tokenize(r["subject"]))
            for tok, cnt in tf_subj.items():
                if tok in w2i:
                    idx = w2i[tok]
                    wt  = math.log1p(cnt)
                    subj_word_f.append((sid, idx))
                    subj_word_r.append((idx, sid))
                    subj_word_w.extend([wt, wt])

    #create graph, with reverse edges
    data_dict = {}
    if include_body_edges:
        data_dict.update({
            ("document", "in_body", "word"): to_tensor(doc_word_f),
            ("word", "rev_in_body", "document"): to_tensor(doc_word_r),
        })
    if include_metadata:
        data_dict.update({
            ("subject", "contains", "word"): to_tensor(subj_word_f),
            ("word", "rev_contains", "subject"): to_tensor(subj_word_r),
            ("document", "has_subject", "subject"): to_tensor(doc_subj_f),
            ("subject", "rev_has_subject", "document"): to_tensor(doc_subj_r),
            ("document", "written_by", "sender"): to_tensor(doc_sender_f),
            ("sender", "rev_written_by", "document"): to_tensor(doc_sender_r),
            ("sender", "affiliated_with", "organization"): to_tensor(sender_org_f),
            ("organization", "rev_affiliated_with", "sender"): to_tensor(sender_org_r),
        })
    g = dgl.heterograph(data_dict)

    #Apply edge weights
    if include_body_edges:
        g.edges["in_body"].data["weight"]     = torch.tensor(doc_word_w[0::2], dtype=torch.float32)
        g.edges["rev_in_body"].data["weight"] = torch.tensor(doc_word_w[1::2], dtype=torch.float32)
    if include_metadata:
        g.edges["contains"].data["weight"]     = torch.tensor(subj_word_w[0::2], dtype=torch.float32)
        g.edges["rev_contains"].data["weight"] = torch.tensor(subj_word_w[1::2], dtype=torch.float32)
    for et in ["has_subject", "rev_has_subject", "written_by", "rev_written_by", "affiliated_with", "rev_affiliated_with"]:
        if et in g.etypes:
            g.edges[et].data["weight"] = torch.ones(g.num_edges(et))

    #Implement the node features (embeddings)
    g.nodes["document"].data["raw_feat"] = torch.tensor(doc_vec)
    g.nodes["word"].data["raw_feat"]     = torch.tensor(W)
    if include_metadata:
        g.nodes["subject"].data["raw_feat"]      = torch.tensor(subj_vec)
        g.nodes["sender"].data["raw_feat"]       = torch.nn.Embedding(len(sender2i), 256)(torch.arange(len(sender2i)))
        g.nodes["organization"].data["raw_feat"] = torch.nn.Embedding(len(org2i), 256)(torch.arange(len(org2i)))

    #Add labels and apply Train/Val/Test masks
    y, classes = pd.factorize(df["newsgroup"])
    g.nodes["document"].data["label"] = torch.tensor(y)
    tr = torch.zeros(len(df), dtype=torch.bool)
    va = torch.zeros(len(df), dtype=torch.bool)
    te = torch.zeros(len(df), dtype=torch.bool)
    tr_r, va_r = 0.6, 0.2
    for c in range(len(classes)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        t = int(n * tr_r)
        v = t + int(n * va_r)
        tr[idx[:t]] = True
        va[idx[t:v]] = True
        te[idx[v:]] = True
    g.nodes["document"].data.update(train_mask=tr, val_mask=va, test_mask=te)
    for n in g.ntypes:
        if n == "document": continue
        z = torch.zeros(g.num_nodes(n), dtype=torch.bool)
        g.nodes[n].data.update(train_mask=z, val_mask=z, test_mask=z,
                               label=torch.full((g.num_nodes(n),), -1))
    return g


#display the graph using graphistry
def display_with_graphistry(hetero_g):
    #register using my key
    graphistry.register(
        api=3,
        protocol="https",
        server="hub.graphistry.com",
        personal_key_id="1OVIVZJHQG",
        personal_key_secret="WDIQ85OCM90TNKS1"
    )

    #convert g nodes into nodes dataframe
    node_rows = []
    for ntype in hetero_g.ntypes:
        count = hetero_g.num_nodes(ntype)
        for nid in range(count):
            node_rows.append({
                'node': f"{ntype}{nid}",  # unique global ID
                'ntype': ntype
            })
    nodes_df = pd.DataFrame(node_rows)

    #convert g edges into edges dataframe
    edge_rows = []
    for srctype, etype, dsttype in hetero_g.canonical_etypes:
        src_ids, dst_ids = hetero_g.edges(etype=(srctype, etype, dsttype))
        for s, d in zip(src_ids.tolist(), dst_ids.tolist()):
            edge_rows.append({
                'src': f"{srctype}{s}",
                'dst': f"{dsttype}{d}",
                'etype': etype,
                'srctype': srctype,
                'dsttype': dsttype
            })
    edges_df = pd.DataFrame(edge_rows)

    #display using graphistry (browser based)
    graphistry.bind(
        source='src',
        destination='dst',
        edge_title='etype',
        node='node'
    ).plot(edges_df, nodes_df)