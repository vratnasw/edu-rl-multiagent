"""Preflight for edu-rl-multiagent.

Checks (in order):
  1. .env loaded (./.env, ../.env, ../edu-data-pipeline/.env)
  2. All 5 R2 env vars present
  3. R2 bucket reachable
  4. Layer 5 world model exists on R2
  5. Layer 4 embeddings exist on R2
  6. PyTorch + CUDA availability (info-only)
  7. Required config keys present

Exits 0 if all critical checks pass.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s :: %(message)s",
                      datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REQUIRED_R2 = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
                  "R2_ENDPOINT_URL")
REQUIRED_PKGS = ('torch', 'numpy', 'pandas', 'yaml', 'boto3')
REQUIRED_CONFIG = {'data': ['parquet_path', 'world_model_checkpoint'], 'agents': ['state_agent', 'county_agent', 'district_agent'], 'communication': ['message_dim', 'n_communication_rounds'], 'training': ['total_steps', 'batch_size', 'lr']}

WORLD_MODEL_KEY = "checkpoints/edu-world-model/best.pt"
EMBEDDINGS_KEY = "embeddings/edu-gnn/district_embeddings.pt"


def _result(name, ok, msg="", critical=True):
    sym = "PASS" if ok else ("FAIL" if critical else "INFO")
    log.info("  %-32s %s  %s", name, sym, msg)
    return {"name": name, "pass": ok, "msg": msg, "critical": critical}


def check_env_loaded():
    tried = []
    for cand in (REPO / ".env", REPO.parent / ".env",
                    REPO.parent / "edu-data-pipeline" / ".env"):
        tried.append(str(cand))
        if cand.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(cand, override=False)
                return _result("env_file", True, f"loaded {cand}")
            except ImportError:
                # Fallback: manual parse
                for line in cand.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
                return _result("env_file", True,
                                  f"loaded {cand} (no python-dotenv)")
    return _result("env_file", False,
                      f"no .env at any of: {tried}")


def check_r2_env():
    missing = [k for k in REQUIRED_R2 if not os.environ.get(k)]
    if missing:
        return _result("r2_env_vars", False, f"missing: {missing}")
    return _result("r2_env_vars", True, "all 5 R2 vars present")


def check_r2_bucket():
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client as r2
        info = r2.smoke_check()
        if info.get("ok"):
            return _result("r2_bucket", True,
                              f"bucket={info.get('bucket')}")
        return _result("r2_bucket", False, f"{info}")
    except Exception as e:  # noqa: BLE001
        return _result("r2_bucket", False, f"exception: {e}")


def check_world_model():
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client as r2
        info = r2.exists(WORLD_MODEL_KEY)
        if info is None:
            return _result("world_model_ckpt", False,
                              f"not found on R2: {WORLD_MODEL_KEY}")
        return _result("world_model_ckpt", True,
                          f"{info['size']:,} bytes at {WORLD_MODEL_KEY}")
    except Exception as e:  # noqa: BLE001
        return _result("world_model_ckpt", False, f"exception: {e}")


def check_embeddings():
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client as r2
        info = r2.exists(EMBEDDINGS_KEY)
        if info is None:
            # Fallback: local on-disk path
            local = (REPO.parent / "edu-gnn" / "results"
                        / "embeddings" / "district_embeddings.pt")
            if local.exists():
                return _result("layer4_embeddings", True,
                                  f"R2 key missing; local fallback at {local}")
            return _result("layer4_embeddings", False,
                              f"not on R2 ({EMBEDDINGS_KEY}) "
                              f"and no local fallback at {local}")
        return _result("layer4_embeddings", True,
                          f"{info['size']:,} bytes at {EMBEDDINGS_KEY}")
    except Exception as e:  # noqa: BLE001
        return _result("layer4_embeddings", False, f"exception: {e}")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return _result("gpu", True,
                              f"{n} CUDA, {name} {total:.1f} GB",
                              critical=False)
        return _result("gpu", True, "CPU only (info)", critical=False)
    except Exception as e:  # noqa: BLE001
        return _result("gpu", False, f"exception: {e}", critical=False)


def check_packages():
    missing = []
    for p in REQUIRED_PKGS:
        try:
            __import__(p)
        except ImportError:
            missing.append(p)
    if missing:
        return _result("python_packages", False,
                          f"missing: {missing}", critical=False)
    return _result("python_packages", True,
                      f"all {len(REQUIRED_PKGS)} importable",
                      critical=False)


def check_config_keys():
    try:
        import yaml
        cfg_path = REPO / "config" / "config.yaml"
        if not cfg_path.exists():
            return _result("config_keys", False, f"missing: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        missing = []
        for section, keys in REQUIRED_CONFIG.items():
            if section not in raw:
                missing.append(section); continue
            if keys:
                for k in keys:
                    if k not in raw[section]:
                        missing.append(f"{section}.{k}")
        if missing:
            return _result("config_keys", False, f"missing: {missing}")
        return _result("config_keys", True,
                          f"all required sections+keys present")
    except Exception as e:  # noqa: BLE001
        return _result("config_keys", False, f"exception: {e}")


def main(strict: bool = True) -> int:
    log.info("=== Preflight: edu-rl-multiagent ===")
    results = [
        check_env_loaded(),
        check_r2_env(),
        check_r2_bucket(),
        check_world_model(),
        check_embeddings(),
        check_packages(),
        check_config_keys(),
        check_gpu(),
    ]
    crit_fail = sum(1 for r in results if not r["pass"] and r["critical"])
    n_pass = sum(1 for r in results if r["pass"])
    log.info("=== %d/%d checks passed (%d critical failures) ===",
                n_pass, len(results), crit_fail)
    if crit_fail and strict:
        log.error("preflight FAILED -- fix critical issues above")
        return 1
    log.info("preflight PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(strict=True))
