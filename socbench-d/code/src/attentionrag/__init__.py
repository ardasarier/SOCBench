"""
From-scratch implementation of AttentionRAG (Fang et al., arXiv:2503.10720).
No public code exists for the paper; this follows Algorithm 1 (Appendix A) and
the prompts in Appendix B.

Deliberately empty of imports. Python executes __init__.py before ANY submodule
import, so re-exporting postprocessor here would make even
`from attentionrag.compression import compress_chunk` load ~2.5GB of Qwen
weights. Import submodules directly instead.

Module demos run as `python -m attentionrag.<module>` from src/ (relative
imports fail under direct script execution).
"""
