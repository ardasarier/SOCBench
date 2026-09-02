"""
AttentionRAG as a LlamaIndex NodePostprocessor.

    retriever.retrieve(query) -> [NodeWithScore]   <- SOCBench measures recall HERE
                              -> AttentionRAGPostprocessor
                              -> compressed [NodeWithScore] -> composition prompt

Benchmarked on all 157 RestBench queries: 2.84x compression, recall
0.509 -> 0.504 (Spotify 1.78x, TMDB 3.42x).

Three consequences of this pipeline position:

1. AttentionRAG is QUERY-AWARE (the hint prefix needs the query), so it cannot
   be one of socrag's CHUNKING_STRATEGIES -- those are NodeParser transforms
   applied at index-build time, before any query exists. Post-retrieval is the
   only structurally valid position.

2. Recall@k cannot improve. SOCBench derives recall from
   node.metadata["endpoints"], which this class preserves untouched. Recall
   only moves downward, when skip_on_none drops a chunk -- the same
   precision-up / recall-down tradeoff Pesl et al. report for the Discovery
   Agent. Evaluation therefore needs compression ratio plus downstream
   composition quality (see scoring.py).

3. LlamaIndex applies postprocessors inside a QueryEngine, not inside
   retriever.retrieve(). The experiment scripts call postprocess_nodes()
   explicitly instead.

Smoke test: python -m attentionrag.postprocessor   (from src/)
"""
from copy import deepcopy
from typing import List, Optional, Tuple

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import PrivateAttr

from .anchor_token import generate_anchor_token
from .attention_features import compute_attention_feature
from .compression import compress_chunk
from .hint_prefix import generate_answer_hint_prefix


class AttentionRAGPostprocessor(BaseNodePostprocessor):
    """Compresses each retrieved node's text using AttentionRAG's Algorithm 1."""

    top_k_tokens: int = 10     # AttentionRAG's k (TOKENS), not SOCBench's k (CHUNKS)
    skip_on_none: bool = True  # Algorithm 1, lines 12-13
    verbose: bool = False
    layer_range: Optional[Tuple[int, int]] = None  # None = all layers (paper default)
    use_anchor: bool = True   # False = skip anchor generation, use the hint's last token
    threshold_ratio: Optional[float] = None   # None = use top_k_tokens instead

    # Algorithm 1 generates the hint once from the query alone (line 5), before
    # chunking. Caching turns 10 LLM calls per query into 1.
    _hint_cache: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        return "AttentionRAGPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            return nodes

        query = query_bundle.query_str

        if query not in self._hint_cache:
            self._hint_cache[query] = generate_answer_hint_prefix(query)
        hint = self._hint_cache[query]

        # No focal token means nothing to guide attention with, so pass through
        # rather than compress badly.
        if hint.strip().lower() == "none":
            if self.verbose:
                print(f"[AttentionRAG] no hint prefix for {query!r}; skipping compression")
            return nodes

        kept: List[NodeWithScore] = []

        for node_with_score in nodes:
            chunk = node_with_score.node.get_content()

            if self.use_anchor:
                anchor = generate_anchor_token(chunk, query, hint)
                if self.skip_on_none and anchor.strip().lower() == "none":
                    if self.verbose:
                        print("[AttentionRAG] dropped chunk (anchor='none')")
                    continue
            else:
                # The prompt already ends with `Answer: {prefix_hint}`, so the
                # final position is the hint's last token. Saves one forward
                # pass per chunk and removes the "none" gate entirely.
                anchor = ""

            token_scores, _ = compute_attention_feature(chunk, query, hint, anchor, layer_range=self.layer_range)

            compressed, _ = compress_chunk(
                chunk, token_scores,
                k=self.top_k_tokens,
                threshold_ratio=self.threshold_ratio,
            )

            if not compressed.strip():
                if self.verbose:
                    print("[AttentionRAG] dropped chunk (nothing survived compression)")
                continue

            # deepcopy: retrieved nodes can be references into the loaded index.
            # Mutating .text in place would leave the index holding compressed
            # text, so later queries in a benchmark loop would see it too.
            new_node = deepcopy(node_with_score.node)
            new_node.text = compressed
            kept.append(NodeWithScore(node=new_node, score=node_with_score.score))

        return kept


if __name__ == "__main__":
    from llama_index.core.schema import TextNode

    # Hand-built node so the interface can be exercised without a FAISS index.
    node = TextNode(
        text=(
            "GET /search: Get Spotify catalog information about albums, artists, "
            "playlists, tracks, shows, episodes or audiobooks that match a keyword string. "
            "Parameters: q (required, string): Your search query. "
            "type (required, string): A comma-separated list of item types to search across. "
            "market (optional, string): An ISO 3166-1 alpha-2 country code. "
            "limit (optional, integer): The maximum number of results to return, default 20. "
            "offset (optional, integer): The index of the first result to return, default 0."
        ),
        metadata={"endpoints": '["GET /search"]'},
    )
    retrieved = [NodeWithScore(node=node, score=0.87)]

    query = "Which parameter controls the maximum number of results returned?"
    postprocessor = AttentionRAGPostprocessor(top_k_tokens=5, verbose=True)
    compressed = postprocessor.postprocess_nodes(retrieved, query_bundle=QueryBundle(query))

    print(f"\nQuery: {query}")
    print(f"\nBefore ({len(retrieved[0].node.get_content())} chars):")
    print(f"  {retrieved[0].node.get_content()!r}")
    print(f"\nAfter ({len(compressed[0].node.get_content())} chars):")
    print(f"  {compressed[0].node.get_content()!r}")
    print(f"\nMetadata preserved: {compressed[0].node.metadata}")
    print(f"Score preserved:    {compressed[0].score}")
