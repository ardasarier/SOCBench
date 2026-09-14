"""
Verifies the environment is ready to work. Loads no models, makes no API calls.

Run from socbench-d/code/:
    python src/check_setup.py
"""
import importlib.util
import os
import sys
from pathlib import Path

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
problems = []


def report(status, label, detail=""):
    print(f"[{status}] {label}" + (f"  --  {detail}" if detail else ""))
    if status == FAIL:
        problems.append(label)


def check_module(name, expected=None):
    """find_spec locates a module without executing it, so this stays fast
    even for torch and transformers."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        report(FAIL, name, "not found")
        return
    version = ""
    if expected:
        try:
            from importlib.metadata import version as get_version
            version = get_version(expected["dist"])
            if version != expected["version"]:
                report(WARN, name, f"{version}, expected {expected['version']}")
                return
        except Exception:
            pass
    report(OK, name, version)


print("\n=== interpreter ===")
major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
report(OK if major_minor == "3.12" else FAIL, "python 3.12", sys.version.split()[0])

in_venv = "venv" in sys.prefix or ".venv" in sys.prefix
report(OK if in_venv else FAIL, "venv active", sys.prefix)

print("\n=== working directory ===")
cwd = Path.cwd()
report(OK if (cwd / "src").is_dir() else FAIL, "running from code/", str(cwd))
report(OK if (cwd / "data").is_dir() else WARN, "data/ exists")

print("\n=== third-party packages ===")
check_module("torch")
check_module("transformers", {"dist": "transformers", "version": "4.47.1"})
check_module("llama_index.core", {"dist": "llama-index-core", "version": "0.12.10"})
check_module("llama_index.embeddings.huggingface")
check_module("llama_index.vector_stores.faiss")
check_module("faiss")
check_module("tiktoken")
check_module("openai", {"dist": "openai", "version": "1.59.6"})
check_module("pydantic")
check_module("sklearn")
check_module("scipy")

print("\n=== socbenchsc (editable install) ===")
check_module("socbenchsc.analysis")

print("\n=== project modules (need cwd = code/) ===")
sys.path.insert(0, str(cwd / "src"))
check_module("socrag.index")
check_module("attentionrag.compression")
check_module("scoring")
# benchmark constructs an openai.Client() at import time, so it needs a key
# present even though nothing calls the API.
if os.environ.get("OPENAI_API_KEY"):
    check_module("benchmark")
else:
    report(WARN, "benchmark", "skipped -- needs OPENAI_API_KEY set to import")

print("\n=== environment variables ===")
for var, needed in [("OPENAI_API_KEY", True), ("HF_TOKEN", False)]:
    value = os.environ.get(var, "")
    if not value:
        report(FAIL if needed else WARN, var, "not set")
    elif value.startswith("not-a-real"):
        report(WARN, var, "still the dummy value -- fine for retrieval, not for codegen")
    else:
        report(OK, var, f"set ({len(value)} chars)")

print("\n=== compute backend ===")
try:
    import torch
    if torch.backends.mps.is_available():
        report(OK, "MPS (Apple GPU)", "available")
    else:
        report(WARN, "MPS", "unavailable -- will run on CPU, ~10x slower")
except Exception as e:
    report(FAIL, "torch backend", str(e))

print("\n=== data ===")
data = cwd / "data"
if data.is_dir():
    index = data / "restbench"
    report(OK if index.is_dir() else WARN, "FAISS index",
           "present" if index.is_dir() else "missing -- rebuilt on first retrieval run")
    contexts = sorted(data.glob("*.json"))
    report(OK if contexts else WARN, f"prepared contexts ({len(contexts)} files)")
    for path in contexts:
        print(f"         {path.name}")

print("\n" + "=" * 60)
if problems:
    print(f"{len(problems)} problem(s): " + ", ".join(problems))
else:
    print("All good. Next: python src/scoring.py, then python src/reproduce_baseline.py")
print("=" * 60)