#R-GCN Definition and Training using 20-Newsgroups
#Developed by: Eliot Turner, Connor Balint, Yuth Yun, Anthony Marcic
#Description: Loads HIN of 20_newsgroups Dataset using "util" functions,
#             and trains a R-GCN using that HIN and tests the model.

# ── Hyper-parameters ──────────────────────────────────────────────────────
DATA_ROOT          = "20_newsgroups"
VOCAB_MIN_DF       = 5
VOCAB_MAX_DF_RATIO = 0.25
SAMPLE_SIZE        = 10000
FULL_TEST          = True
GRAPH_MODE         = 'words'     #{'full','words','metadata'}
HIDDEN_DIM         = 256
NUM_LAYERS         = 3
DROPOUT            = 0.1
LR, WEIGHT_DECAY   = 1e-3, 5e-4
NUM_EPOCHS         = 35
BATCH_SIZE         = 128     # None → full-batch
PLOT_FILE          = "accuracy_curve.png"
METRICS_FILE       = "metrics.csv"
CONF_MATRIX_FILE   = "confusion_matrix.csv"
# ──────────────────────────────────────────────────────────────────────────

#Necessary Libraries
import time
import torch, dgl, matplotlib.pyplot as plt, torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import utils

#Use GPU if possible for large HIN
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load a random sample or full 20_newsgroups to a dataframe (dropping samples w/o metadata)
df, drop, tot = utils.build_dataframe(DATA_ROOT, SAMPLE_SIZE, FULL_TEST)
#Compute IDF weights, and trim vocabulary based on parameters
idf, vocab    = utils.compute_idf(df, VOCAB_MIN_DF, VOCAB_MAX_DF_RATIO)
#convert DF to DGL Heterograph HIN
g             = utils.build_heterograph(df, idf, vocab, graph_mode=GRAPH_MODE).to(DEV)
print(f"Graph: {g.num_nodes()} nodes, {g.num_edges()} edges | dropped {drop}/{tot}")

#optional display function to visualize HIN (unstable for >10,000 SAMPLE_SIZE)
#utils.display_with_graphistry(g)

#number of classes (should be 20, but can vary if using small sample size)
NCLS = df["newsgroup"].nunique()

#Model Definition
class RGCN(torch.nn.Module):
    #optional types for metadata-less processing
    OPTIONAL_NTYPES = ("subject", "sender", "organization")

    def __init__(self, h, n_cls, etypes, n_layers = 2, p=0.3):
        super().__init__()
        #3 Linear layers for each embedding type (GloVe, BERT, learned)
        self.pw = torch.nn.Linear(300, h, bias=False)   # word
        self.pd = torch.nn.Linear(512, h, bias=False)   # document
        self.pm = torch.nn.Linear(256, h, bias=False)   # metadata

        #Convolutional Layers (controllable by hyperparameter)
        self.convs = torch.nn.ModuleList([
            #each layer applies GraphConv on each edge type
            dgl.nn.HeteroGraphConv(
                {e: dgl.nn.GraphConv(h, h, norm="right") for e in etypes},
                aggregate="sum")
            for _ in range(n_layers)
        ])

        #Linear layer
        self.cls  = torch.nn.Linear(h, n_cls)
        #ReLU activation and dropout layers
        self.act, self.drop = torch.nn.ReLU(), torch.nn.Dropout(p)

    def forward(self, g):
        with g.local_scope():
            #linear layers for default node types
            h = {
                "word":     self.pw(g.nodes["word"].data["raw_feat"]),
                "document": self.pd(g.nodes["document"].data["raw_feat"]),
            }
            #linear layers for optional node types
            for nt in self.OPTIONAL_NTYPES:
                if nt in g.ntypes:
                    h[nt] = self.pm(g.nodes[nt].data["raw_feat"])
            #apply activation and dropout after every convolution layer
            for conv in self.convs:
                h = {k: self.drop(self.act(v)) for k, v in conv(g, h).items()}
            return self.cls(h["document"])

#instantiate model using hyperparams
auto  = RGCN(HIDDEN_DIM, NCLS, g.etypes, NUM_LAYERS, DROPOUT).to(DEV)

#instantiate optimizer (AdamW) using hyperparams
optim = torch.optim.AdamW(auto.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

#Data Loaders
def make_loader(split, bs):
    mask  = g.nodes["document"].data[f"{split}_mask"]
    seeds = g.nodes("document")[mask]
    #Batches data
    if bs is None:
        return [(seeds, g)]
    #sampled NUM_LAYERS hops from seed, for full functionality
    sampler = MultiLayerFullNeighborSampler(NUM_LAYERS)
    return DataLoader(g, {"document": seeds}, sampler,
                      batch_size=bs, shuffle=(split == "train"))

#loaders for all sets
train_loader = make_loader("train", BATCH_SIZE)
val_loader   = make_loader("val",   BATCH_SIZE)
test_loader  = make_loader("test",  BATCH_SIZE)

#Some helper functions
#unpacks batch, loads to device, returns subgraph
def unpack(batch):
    if isinstance(batch[1], dgl.DGLGraph):
        return batch[1].to(DEV)
    in_dict, out_dict, _ = batch
    node_dict = {}
    for d in (in_dict, out_dict):
        for nt, nids in d.items():
            node_dict.setdefault(nt, []).append(nids)
    node_dict = {nt: torch.unique(torch.cat(v)).to(DEV) for nt, v in node_dict.items()}
    return dgl.node_subgraph(g, node_dict)

#purges tensors on device to avoid GPU VRAM overloading
def purge_autograd_tensors(graph):
    for nt in graph.ntypes:
        for k in list(graph.nodes[nt].data.keys()):
            t = graph.nodes[nt].data[k]
            if isinstance(t, torch.Tensor) and t.requires_grad:
                graph.nodes[nt].data[k] = t.detach()
    for et in graph.canonical_etypes:
        for k in list(graph.edges[et].data.keys()):
            t = graph.edges[et].data[k]
            if isinstance(t, torch.Tensor) and t.requires_grad:
                graph.edges[et].data[k] = t.detach()

#Training/Running Function
def run(loader, split, train=False):
    auto.train() if train else auto.eval()
    #variable to save results
    loss_sum, correct, total = 0., 0, 0
    for batch in loader:
        #unpack to subgraph
        sg = unpack(batch)
        #run model
        logits = auto(sg)
        #collect documents and labels
        m     = sg.nodes["document"].data[f"{split}_mask"].to(DEV)
        y     = sg.nodes["document"].data["label"][m]
        pred  = logits[m].argmax(1)
        #find loss and correct rate
        correct += (pred == y).sum().item(); total += y.numel()
        loss = F.cross_entropy(logits[m], y)
        if train:
            #update grad
            optim.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            purge_autograd_tensors(sg)
            optim.step()
        #update vars
        loss_sum += loss.item()
    return loss_sum / max(1, len(loader)), (correct / total if total else 0.)


#Training Loop
#values to save and print
tr_a, va_a, te_a = [], [], []
tr_l, va_l, te_l = [], [], []
best_test_acc = -1.
best_preds = None
best_labels = None
print(f"Training on {DEV} …")

#iterate epochs
for ep in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    #run all loaders
    tl, ta    = run(train_loader, "train", True)
    vl, va    = run(val_loader,   "val",   False)
    testl, te = run(test_loader,  "test",  False)
    t1 = time.time() #collect duration

    #append the data from that run
    tr_a.append(ta); va_a.append(va); te_a.append(te)
    tr_l.append(tl); va_l.append(vl); te_l.append(testl)

    #track best performance, and save test data for confusion matrix
    if te > best_test_acc:
        best_test_acc = te
        auto.eval()
        node_preds = {}
        node_labels = {}
        with torch.no_grad():
            for batch in test_loader:
                sg = unpack(batch)
                doc_ids = sg.nodes["document"].data[dgl.NID]  # Unique global node IDs
                m = sg.nodes["document"].data["test_mask"].to(DEV)
                logits = auto(sg)
                labs = sg.nodes["document"].data["label"][m]
                preds = logits[m].argmax(1)

                for nid, pred, lab in zip(doc_ids[m.cpu()], preds.cpu(), labs.cpu()):
                    node_preds[nid.item()] = pred
                    node_labels[nid.item()] = lab

        # Sort by node ID to ensure consistent ordering
        sorted_ids = sorted(node_preds.keys())
        best_preds = torch.tensor([node_preds[i] for i in sorted_ids])
        best_labels = torch.tensor([node_labels[i] for i in sorted_ids])

    print(f"Ep {ep:02d}/{NUM_EPOCHS} │ loss {tl:.4f} │ train {ta:.3f} │ val {va:.3f} │ test {te:.3f} │ {t1 - t0:.1f}s")

#Create a CSV for futher plotting
metrics_df = pd.DataFrame({
    "epoch":      list(range(1, NUM_EPOCHS + 1)),
    "train_loss": tr_l,
    "val_loss":   va_l,
    "test_loss":  te_l,
    "train_acc":  tr_a,
    "val_acc":    va_a,
    "test_acc":   te_a,
})
metrics_df.to_csv(METRICS_FILE, index=False)
print(f"Saved {METRICS_FILE}")

#Create confusion matrix of best epoch
cm = confusion_matrix(best_labels, best_preds)
doc_labels = g.nodes["document"].data["label"].cpu().numpy()
newsgroups = df["newsgroup"].values
label2name  = {}
for code, name in zip(doc_labels, newsgroups):
    label2name.setdefault(code, name)
class_names = [label2name[i] for i in range(NCLS)]
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(CONF_MATRIX_FILE)
print(f"Saved {CONF_MATRIX_FILE} (for best test epoch)")

#plot the train/val/test accuracies over epoch
plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), tr_a, label="Train")
plt.plot(range(1, NUM_EPOCHS + 1), va_a, label="Val")
plt.plot(range(1, NUM_EPOCHS + 1), te_a, label="Test")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("R-GCN accuracy")
plt.ylim(0, 1)
plt.legend(); plt.grid(); plt.savefig(PLOT_FILE, dpi=160)
print(f"Saved {PLOT_FILE}")