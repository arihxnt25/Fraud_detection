"""Heterogeneous fraud-network graph.

Nodes:  card, device, email_payer, email_recv, address, ip_proxy (+ optional
        transaction nodes for investigation sub-graphs)
Edges:  card --uses--> device / email / ip_proxy, card --occurs_at--> address,
        card --connected_through--> card (projection via shared entities)

Why a graph? A single transaction from a fresh card looks clean to a tabular
model. But if that card shares a device and billing address with five cards
already charged back, it is almost certainly part of a fraud ring. The
``FraudGraph`` turns that structure into numeric features and ranked clusters.

Leakage control: labels are only ever read from the data passed to ``fit``
(the training window), and a card's own label is excluded (leave-one-out)
when computing its neighbourhood fraud rate.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from fraudnet.config import ENTITY_COLUMNS, TARGET
from fraudnet.features.engineering import add_card_uid

log = logging.getLogger(__name__)

EDGE_TYPE = {"device": "uses", "email_payer": "uses", "email_recv": "uses",
             "ip_proxy": "uses", "address": "occurs_at"}


def node_id(kind: str, value) -> str:
    return f"{kind}:{value}"


@dataclass
class FraudCluster:
    cluster_id: int
    n_cards: int
    n_fraud_cards: int
    fraud_rate: float
    n_transactions: int
    total_amount: float
    shared_entities: list[str]
    cards: list[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class FraudGraph:
    def __init__(self, max_entity_degree: int | None = None, min_entity_degree: int = 2,
                 smoothing: float = 5.0, min_shared: int = 2):
        """
        Args:
            max_entity_degree: entities linked to more cards than this are hubs
                (e.g. ``gmail.com``, a popular phone model) and are ignored for
                ring linking. ``None`` = adaptive: 1% of cards, clipped to
                [100, 1000], so the cutoff scales from a 10% sample to full data.
            min_entity_degree: entities seen with fewer cards carry no link info.
        """
        self.max_entity_degree = max_entity_degree
        self.max_degree_: int = max_entity_degree or 100
        self.smoothing = smoothing  # pseudo-cards pulling small groups to the prior
        self.prior: float = 0.0
        self.min_entity_degree = min_entity_degree
        self.min_shared = min_shared
        self.entity_component: dict[str, int] = {}
        self._links: pd.DataFrame | None = None
        self.G = nx.Graph()
        self.card_stats: pd.DataFrame | None = None
        self.entity_stats: pd.DataFrame | None = None
        self.component_of: dict[str, int] = {}
        self.members: dict[int, list[str]] = {}
        self.component_stats: pd.DataFrame | None = None
        self.proximity: dict[str, float] = {}

    # ------------------------------------------------------------------ build
    @staticmethod
    def _entity_long(df: pd.DataFrame) -> pd.DataFrame:
        """Long table of (card_uid, entity_node, kind) links."""
        frames = []
        for kind, cols in ENTITY_COLUMNS.items():
            if kind == "card":
                continue
            col = cols[0]
            if col not in df:
                continue
            sub = df[["card_uid", col]].dropna()
            frames.append(pd.DataFrame({"card_uid": sub["card_uid"].values,
                                        "entity": [node_id(kind, v) for v in sub[col].values],
                                        "kind": kind}))
        return pd.concat(frames, ignore_index=True) if frames else \
            pd.DataFrame(columns=["card_uid", "entity", "kind"])

    def fit(self, df: pd.DataFrame) -> FraudGraph:
        if "card_uid" not in df:
            df = add_card_uid(df)
        has_label = TARGET in df

        cards = df.groupby("card_uid").agg(
            n_tx=("TransactionID", "size"),
            amount=("TransactionAmt", "sum"),
            fraud=(TARGET, "max") if has_label else ("TransactionAmt", lambda s: 0),
        )
        self.card_stats = cards
        self.prior = float(cards["fraud"].mean()) if len(cards) else 0.0
        self.max_degree_ = self.max_entity_degree or int(np.clip(0.01 * len(cards), 100, 1000))

        links = self._entity_long(df)
        link_counts = links.groupby(["card_uid", "entity", "kind"]).size().rename("weight").reset_index()
        link_counts = link_counts.join(cards["fraud"], on="card_uid")
        self._links = link_counts

        ent = link_counts.groupby(["entity", "kind"]).agg(
            n_cards=("card_uid", "nunique"), n_fraud_cards=("fraud", "sum"),
            n_tx=("weight", "sum")).reset_index().set_index("entity")
        self.entity_stats = ent

        # --- graph: card nodes, entity nodes, typed edges ---------------------
        G = nx.Graph()
        for uid, row in cards.iterrows():
            G.add_node(node_id("card", uid), kind="card", n_tx=int(row.n_tx),
                       amount=float(row.amount), fraud=int(row.fraud))
        for e, row in ent.iterrows():
            G.add_node(e, kind=row.kind, n_cards=int(row.n_cards),
                       n_fraud_cards=int(row.n_fraud_cards),
                       hub=bool(row.n_cards > self.max_degree_))
        G.add_edges_from(
            (node_id("card", r.card_uid), r.entity,
             {"relation": EDGE_TYPE.get(r.kind, "uses"), "weight": int(r.weight)})
            for r in link_counts.itertuples(index=False))
        self.G = G

        self._compute_components()
        self._compute_proximity()
        self._links = None
        log.info("Graph: %d nodes, %d edges, %d multi-card clusters",
                 G.number_of_nodes(), G.number_of_edges(),
                 int((self.component_stats["n_cards"] > 1).sum()))
        return self

    def _linking_subgraph(self) -> nx.Graph:
        keep = [n for n, d in self.G.nodes(data=True)
                if d["kind"] == "card" or
                (self.min_entity_degree <= d.get("n_cards", 0) <= self.max_degree_)]
        return self.G.subgraph(keep)

    def _compute_components(self) -> None:
        """Fraud-ring clusters on the card--card projection.

        Two cards are linked (``connected_through``) only if they share at
        least ``min_shared`` distinct non-hub entities, e.g. the same device
        *and* the same billing address. One shared attribute (a popular phone
        model, a big e-mail provider) is coincidence; several is a ring.
        """
        lk = self._links
        lk = lk[lk.entity.map(self.entity_stats["n_cards"]).between(
            self.min_entity_degree, self.max_degree_)][["card_uid", "entity"]]
        pairs = lk.merge(lk, on="entity")
        pairs = pairs[pairs.card_uid_x < pairs.card_uid_y]
        shared = pairs.groupby(["card_uid_x", "card_uid_y"]).size()
        shared = shared[shared >= self.min_shared].reset_index(name="n_shared")

        C = nx.Graph()
        C.add_nodes_from(node_id("card", u) for u in self.card_stats.index)
        edges = [(node_id("card", a), node_id("card", b), {"relation": "connected_through",
                                                           "weight": int(w)})
                 for a, b, w in shared.itertuples(index=False)]
        C.add_edges_from(edges)
        self.G.add_edges_from(edges)

        self.component_of, self.members, rows = {}, {}, []
        for cid, comp in enumerate(sorted(nx.connected_components(C), key=len, reverse=True)):
            cards = list(comp)
            for n in cards:
                self.component_of[n] = cid
            self.members[cid] = cards
            rows.append({"component": cid, "n_cards": len(cards),
                         "n_fraud_cards": sum(self.G.nodes[n]["fraud"] for n in cards),
                         "n_tx": sum(self.G.nodes[n]["n_tx"] for n in cards),
                         "amount": sum(self.G.nodes[n]["amount"] for n in cards)})
        cs = pd.DataFrame(rows).set_index("component")
        cs["fraud_rate"] = cs["n_fraud_cards"] / cs["n_cards"].clip(lower=1)

        # attach each non-hub entity to the component holding most of its cards,
        # so an unseen card can be matched to an existing ring via its entities
        lk = lk.assign(comp=lk.card_uid.map(lambda u: self.component_of[node_id("card", u)]))
        multi = lk[lk.comp.map(cs["n_cards"]) > 1]
        ent_comp = multi.groupby("entity")["comp"].agg(lambda c: c.value_counts().index[0])
        self.entity_component = ent_comp.to_dict()
        for e, cid in self.entity_component.items():
            self.members[cid].append(e)
        cs["n_entities"] = pd.Series(ent_comp.value_counts()).reindex(cs.index).fillna(0).astype(int)
        self.component_stats = cs

    def _compute_proximity(self) -> None:
        """Personalised PageRank seeded on known fraud cards = 'fraud proximity'."""
        H = self._linking_subgraph()
        seeds = {n: 1.0 for n, d in H.nodes(data=True) if d.get("fraud") == 1}
        if not seeds or H.number_of_edges() == 0:
            self.proximity = {}
            return
        pr = nx.pagerank(H, alpha=0.85, personalization=seeds, max_iter=100, tol=1e-6)
        vals = np.array(list(pr.values()))
        scale = np.percentile(vals, 99.5) or vals.max() or 1.0
        self.proximity = {k: float(min(v / scale, 1.0)) for k, v in pr.items()}

    # --------------------------------------------------------------- features
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-transaction graph features (works for unseen cards too)."""
        if self.entity_stats is None:
            raise RuntimeError("FraudGraph must be fit first")
        if "card_uid" not in df:
            df = add_card_uid(df)
        out = pd.DataFrame(index=df.index)
        card_fraud = self.card_stats["fraud"] if self.card_stats is not None else pd.Series(dtype=float)
        own_fraud = df["card_uid"].map(card_fraud).fillna(0).values
        seen = df["card_uid"].isin(card_fraud.index).values

        es = self.entity_stats
        max_rates, max_counts = [], []
        for kind, cols in ENTITY_COLUMNS.items():
            if kind == "card" or cols[0] not in df:
                continue
            nodes = df[cols[0]].map(lambda v, k=kind: node_id(k, v) if pd.notna(v) else None)
            n_cards = nodes.map(es["n_cards"]).fillna(0).values
            n_fraud = nodes.map(es["n_fraud_cards"]).fillna(0).values
            # leave-one-out: remove the card's own contribution if it was in training
            loo_cards = np.where(seen & (n_cards > 0), n_cards - 1, n_cards)
            loo_fraud = np.where(seen & (n_cards > 0), n_fraud - own_fraud, n_fraud)
            rate = self._smooth(loo_fraud, loo_cards)
            out[f"g_{kind}_n_cards"] = n_cards
            out[f"g_{kind}_fraud_rate"] = rate
            max_rates.append(rate)
            max_counts.append(n_cards)
        if max_rates:
            out["g_max_entity_fraud_rate"] = np.max(max_rates, axis=0)
            out["g_max_entity_n_cards"] = np.max(max_counts, axis=0)

        comp_size, comp_rate, prox = [], [], []
        comp_rates = self.component_stats["fraud_rate"].to_dict()
        comp_n = self.component_stats["n_cards"].to_dict()
        comp_f = self.component_stats["n_fraud_cards"].to_dict()
        for uid, own, s, row in zip(df["card_uid"].values, own_fraud, seen,
                                    df.itertuples(index=False)):
            cid = self.component_of.get(node_id("card", uid))
            if cid is None:  # unseen card: join a ring if it shares >= min_shared of its entities
                votes = Counter(c for c in (self.entity_component.get(e)
                                            for e in self._row_entities(row)) if c is not None)
                ok = [c for c, v in votes.items() if v >= self.min_shared]
                cid = max(ok, key=lambda c: comp_rates[c]) if ok else None
            if cid is None:
                comp_size.append(0); comp_rate.append(self.prior); prox.append(0.0)
                continue
            n, f = comp_n[cid] - (1 if s else 0), comp_f[cid] - (own if s else 0)
            comp_size.append(int(comp_n[cid]))
            comp_rate.append(float(self._smooth(f, n)))
            prox.append(self.proximity.get(node_id("card", uid), 0.0) if s else
                        max((self.proximity.get(e, 0.0) for e in self._row_entities(row)), default=0.0))
        out["g_component_size"] = comp_size
        out["g_component_fraud_rate"] = comp_rate
        out["g_fraud_proximity"] = prox
        return out.astype(np.float32)

    def _smooth(self, n_fraud, n_cards):
        """Bayesian-smoothed fraud rate: one fraud card out of one is not 100% risk."""
        return (n_fraud + self.prior * self.smoothing) / (n_cards + self.smoothing)

    @staticmethod
    def _row_entities(row) -> list[str]:
        ents = []
        for kind, cols in ENTITY_COLUMNS.items():
            v = getattr(row, cols[0], None) if kind != "card" else None
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and v is not pd.NA:
                ents.append(node_id(kind, v))
        return ents

    # ----------------------------------------------------------- investigation
    def fraud_clusters(self, min_cards: int = 3, min_fraud_rate: float = 0.3,
                       top_k: int = 50) -> list[FraudCluster]:
        cs = self.component_stats
        cand = cs[(cs.n_cards >= min_cards) & (cs.fraud_rate >= min_fraud_rate)]
        cand = cand.sort_values(["n_fraud_cards", "fraud_rate"], ascending=False).head(top_k)
        out = []
        for cid, st in cand.iterrows():
            members = self.members[cid]
            out.append(FraudCluster(
                cluster_id=int(cid), n_cards=int(st.n_cards), n_fraud_cards=int(st.n_fraud_cards),
                fraud_rate=round(float(st.fraud_rate), 4), n_transactions=int(st.n_tx),
                total_amount=round(float(st.amount), 2),
                shared_entities=sorted(m for m in members if not m.startswith("card:")),
                cards=sorted(m for m in members if m.startswith("card:"))))
        return out

    def subgraph(self, node: str, radius: int = 2, max_nodes: int = 300) -> dict:
        """Ego network around a node, serialised for the API / dashboard."""
        if node not in self.G:
            raise KeyError(node)
        H = self._linking_subgraph()
        center_graph = H if node in H else self.G
        ego = nx.ego_graph(center_graph, node, radius=radius)
        if ego.number_of_nodes() > max_nodes:
            keep = sorted(ego.nodes, key=lambda n: nx.shortest_path_length(ego, node, n))[:max_nodes]
            ego = ego.subgraph(keep)
        return {
            "center": node,
            "nodes": [{"id": n, **{k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()},
                       "proximity": round(self.proximity.get(n, 0.0), 4)} for n, d in ego.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in ego.edges(data=True)],
        }

    def summary(self) -> dict:
        cs = self.component_stats
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(),
                "cards": int((cs.n_cards).sum()) if cs is not None else 0,
                "multi_card_clusters": int((cs.n_cards > 1).sum()) if cs is not None else 0,
                "suspicious_clusters": len(self.fraud_clusters())}
