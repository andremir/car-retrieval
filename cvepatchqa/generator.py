"""
CVE/Security-Patch Domain Generator
=====================================
Produces TC-MQA examples for the CVE/patch domain.

Entity mapping:
  employee_id     → product_id  (e.g. "nginx", "openssl")
  security_ticker → cve_id      (e.g. "CVE-2024-00042")

Hop types generated:
  T0: One CVE_DISCLOSED, no patch. Query: "still affected?" Answer: vulnerable.
  T1: CVE_DISCLOSED → PATCH_RELEASED. Query: "still affected?" Answer: patched.
      KEY: patch vocabulary is disjoint from query → BM25/Dense T1 TCA = 0.

Output format: same JSONL schema as finsuperqa_v1.jsonl.
"""

from __future__ import annotations
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

try:
    from cvepatchqa.rules import EventType
except ImportError:
    from rules import EventType

# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------

PRODUCTS = [
    "nginx", "apache-httpd", "openssl", "log4j", "redis", "mongodb",
    "elasticsearch", "tomcat", "spring-framework", "hibernate",
    "django", "flask", "rails", "wordpress", "drupal",
    "postgresql", "mysql", "sqlite", "memcached", "haproxy",
    "vsftpd", "proftpd", "dovecot", "postfix", "bind9",
    "samba", "openssh", "curl", "wget", "libssl",
    "libuuid", "libpng", "libjpeg", "zlib", "expat",
    "glibc", "libxml2", "libxslt", "pycrypto", "bcrypt",
    "jackson-databind", "lodash", "moment-js", "jquery",
    "react", "angular", "vue-js", "express-js", "koa",
    "fastapi", "aiohttp", "grpc", "protobuf", "thrift",
    "kafka", "rabbitmq", "activemq", "zookeeper", "consul",
    "etcd", "vault", "terraform", "ansible", "puppet",
]

VENDORS = {
    "nginx": "nginx Inc.",
    "apache-httpd": "Apache Software Foundation",
    "openssl": "OpenSSL Project",
    "log4j": "Apache Software Foundation",
    "redis": "Redis Ltd.",
    "mongodb": "MongoDB Inc.",
    "elasticsearch": "Elastic N.V.",
    "tomcat": "Apache Software Foundation",
    "spring-framework": "VMware Inc.",
    "hibernate": "Red Hat Inc.",
    "django": "Django Software Foundation",
    "flask": "Pallets Projects",
    "rails": "Rails Core Team",
    "wordpress": "Automattic Inc.",
    "drupal": "Drupal Association",
    "postgresql": "PostgreSQL Global Development Group",
    "mysql": "Oracle Corporation",
    "sqlite": "Hwaci",
    "memcached": "Memcached Contributors",
    "haproxy": "HAProxy Technologies",
}

SEVERITIES = ["Critical", "High", "Medium"]

ATTACK_VECTORS = [
    "remote code execution",
    "SQL injection",
    "buffer overflow",
    "path traversal",
    "cross-site scripting",
    "XML external entity injection",
    "server-side request forgery",
    "deserialization of untrusted data",
    "privilege escalation",
    "denial of service",
]

CWE_IDS = [
    "CWE-79", "CWE-89", "CWE-119", "CWE-200", "CWE-352",
    "CWE-416", "CWE-611", "CWE-502", "CWE-22", "CWE-918",
]

VERSIONS = ["1.2.{n}", "2.{n}.0", "3.{n}.1", "4.0.{n}", "0.{n}.3"]

_base_time = datetime(2024, 1, 10)


def _cve_id(idx: int) -> str:
    return f"CVE-2024-{idx:05d}"


def _version(rng: random.Random, product: str, bump: int = 0) -> str:
    n = rng.randint(10, 99)
    return f"{n + bump}.{rng.randint(0, 9)}"


def _vendor(product: str) -> str:
    return VENDORS.get(product, f"{product.capitalize()} Maintainers")


# ---------------------------------------------------------------------------
# Natural-language templates — vocabulary gap is intentional
# ---------------------------------------------------------------------------

def _disclosure_text(cve_id: str, product: str, severity: str,
                     attack_vector: str, cwe: str, version: str) -> str:
    vendor = _vendor(product)
    return (
        f"{cve_id} has been identified in {product} (maintained by {vendor}). "
        f"The flaw permits {attack_vector} by unauthenticated remote attackers. "
        f"Severity: {severity} (CVSS 9.x). Weakness: {cwe}. "
        f"Affected versions: {product} up to and including {version}. "
        f"No patch is currently available. Exploitation has been observed in the wild. "
        f"This vulnerability remains unmitigated at time of publication."
    )


def _patch_text(patched_version: str, rng: random.Random) -> str:
    # Deliberately omits product name AND CVE ID to maximize vocabulary gap.
    # Entity identity is carried in Event.security_ticker (cve_id) and
    # Event.employee_id (product_id) — NOT in the text. This matches the
    # real-world scenario where a vendor changelog entry doesn't repeat the
    # CVE identifier and uses internal release notes language.
    adj = rng.choice(["maintenance", "security", "emergency", "critical"])
    action = rng.choice(["addresses", "resolves", "mitigates", "remediates"])
    return (
        f"Release notice: The maintainer team has published version {patched_version} "
        f"as a {adj} update. This release {action} recently disclosed security concerns "
        f"raised by external researchers. "
        f"All installations running prior versions should apply this update promptly. "
        f"Detailed release notes are available in the project changelog."
    )


def _patch_superseded_text(product: str, newer_version: str) -> str:
    vendor = _vendor(product)
    return (
        f"Updated security release: {vendor} has published {product} version {newer_version}, "
        f"superseding the previous security update. "
        f"The prior release did not fully address the underlying weakness. "
        f"The latest build includes additional hardening measures. "
        f"Operators should upgrade to this latest release immediately."
    )


def _workaround_text(cve_id: str, product: str) -> str:
    return (
        f"Interim guidance for {cve_id} in {product}: "
        f"While a permanent fix is not yet available, a temporary control has been issued. "
        f"Apply the following network-level restriction or configuration change to limit exposure. "
        f"This workaround reduces but does not eliminate risk. Monitor for further advisories."
    )


def _query(product: str, cve_id: str) -> str:
    return (
        f"Is {product} still affected by {cve_id}? "
        f"Has this security flaw been addressed and is the system still at risk?"
    )


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _t0_scenario(idx: int, product: str, cve_id: str, rng: random.Random,
                 example_id: str) -> dict:
    """T0: Single CVE_DISCLOSED, no supersession. Answer: vulnerable."""
    severity = rng.choice(SEVERITIES)
    av = rng.choice(ATTACK_VECTORS)
    cwe = rng.choice(CWE_IDS)
    ver = _version(rng, product)
    t0 = _base_time + timedelta(days=idx * 2)

    disclosure_id = str(uuid.uuid4())[:8]
    return {
        "example_id": example_id,
        "domain": "cve_security_patch",
        "hop_type": "type0_no_supersession",
        "employee": product,        # repurposed: product → employee_id
        "ticker": cve_id,           # repurposed: cve_id → security_ticker
        "query": _query(product, cve_id),
        "answer": "vulnerable",
        "kb": [
            {
                "event_id": disclosure_id,
                "event_type": EventType.CVE_DISCLOSED.value,
                "timestamp": t0.isoformat(),
                "text": _disclosure_text(cve_id, product, severity, av, cwe, ver),
            }
        ],
        "supersession_edges": [],
    }


def _t1_scenario(idx: int, product: str, cve_id: str, rng: random.Random,
                 example_id: str) -> dict:
    """T1: CVE_DISCLOSED → PATCH_RELEASED. Answer: patched.
    Vocabulary gap: patch text avoids 'CVE-XXXX', 'vulnerability', 'flaw'.
    BM25/Dense retrieve disclosure (high relevance) but miss patch (low relevance).
    """
    severity = rng.choice(SEVERITIES)
    av = rng.choice(ATTACK_VECTORS)
    cwe = rng.choice(CWE_IDS)
    ver = _version(rng, product)
    patched_ver = _version(rng, product, bump=1)
    t0 = _base_time + timedelta(days=idx * 2)
    t1 = t0 + timedelta(days=rng.randint(3, 14))

    disclosure_id = str(uuid.uuid4())[:8]
    patch_id = str(uuid.uuid4())[:8]

    # T1: patch is the gold superseder, not the trigger in the query
    # Supersession edge: patch → disclosure
    return {
        "example_id": example_id,
        "domain": "cve_security_patch",
        "hop_type": "type1_patch_supersedes_disclosure",
        "employee": product,
        "ticker": cve_id,
        "query": _query(product, cve_id),
        "answer": "patched",
        "kb": [
            {
                "event_id": disclosure_id,
                "event_type": EventType.CVE_DISCLOSED.value,
                "timestamp": t0.isoformat(),
                "text": _disclosure_text(cve_id, product, severity, av, cwe, ver),
            },
            {
                "event_id": patch_id,
                "event_type": EventType.PATCH_RELEASED.value,
                "timestamp": t1.isoformat(),
                "text": _patch_text(patched_ver, rng),
            },
        ],
        "supersession_edges": [
            {"trigger_id": patch_id, "target_id": disclosure_id, "rule": "R_CVE1"},
        ],
    }


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_t0: int = 250,
    n_t1: int = 250,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    # Build unique (product, cve_id) pairs
    product_pool = PRODUCTS[:]
    rng.shuffle(product_pool)
    cve_pairs = []
    cve_counter = 1
    while len(cve_pairs) < n_t0 + n_t1:
        product = product_pool[len(cve_pairs) % len(product_pool)]
        cve_id = _cve_id(cve_counter)
        cve_counter += 1
        cve_pairs.append((product, cve_id))

    examples = []
    idx = 0

    for i in range(n_t0):
        product, cve_id = cve_pairs[idx]
        idx += 1
        eid = f"cve-t0-{i:04d}"
        examples.append(_t0_scenario(i, product, cve_id, rng, eid))

    for i in range(n_t1):
        product, cve_id = cve_pairs[idx]
        idx += 1
        eid = f"cve-t1-{i:04d}"
        examples.append(_t1_scenario(i, product, cve_id, rng, eid))

    return examples


def build_distractor_events(
    examples: list[dict],
    n_distractors: int = 10,
    rng: random.Random = None,
) -> list[dict]:
    """
    Generate distractor events for the shared corpus.
    Three modes (mirrors finsuperqa/corpus.py):
      0: same product, different CVE
      1: different product, same CVE
      2: completely unrelated
    Only CVE_DISCLOSED distractor events (no PATCH_RELEASED) to avoid
    accidentally creating valid supersession chains from distractors.
    """
    if rng is None:
        rng = random.Random(99)

    distractors = []
    all_products = PRODUCTS[:]
    all_cveid_nums = list(range(90000, 99999))

    for ex in examples:
        product = ex["employee"]
        cve_id = ex["ticker"]
        t0 = datetime.fromisoformat(ex["kb"][0]["timestamp"])

        for d_idx in range(n_distractors):
            choice = d_idx % 3
            if choice == 0:
                d_product = product
                d_num = rng.choice(all_cveid_nums)
                d_cve = f"CVE-2023-{d_num:05d}"
            elif choice == 1:
                d_product = rng.choice([p for p in all_products if p != product])
                d_cve = cve_id
            else:
                d_product = rng.choice(all_products)
                d_cve = f"CVE-2023-{rng.choice(all_cveid_nums):05d}"

            severity = rng.choice(SEVERITIES)
            av = rng.choice(ATTACK_VECTORS)
            cwe = rng.choice(CWE_IDS)
            ver = f"{rng.randint(1, 9)}.{rng.randint(0, 9)}"
            d_time = t0 + timedelta(days=rng.randint(-10, 10))

            distractors.append({
                "event_id": str(uuid.uuid4())[:8],
                "event_type": EventType.CVE_DISCLOSED.value,
                "timestamp": d_time.isoformat(),
                "text": _disclosure_text(d_cve, d_product, severity, av, cwe, ver),
                "employee": d_product,
                "ticker": d_cve,
                "session_id": f"distract-{uuid.uuid4().hex[:6]}",
                "is_distractor": True,
            })

    return distractors


if __name__ == "__main__":
    examples = generate_dataset(n_t0=250, n_t1=250)
    out_path = Path(__file__).parent.parent / "data" / "cvepatchqa_v1.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(examples)} examples → {out_path}")
    t_counts = {}
    for ex in examples:
        t_counts[ex["hop_type"]] = t_counts.get(ex["hop_type"], 0) + 1
    for k, v in sorted(t_counts.items()):
        print(f"  {k}: {v}")
