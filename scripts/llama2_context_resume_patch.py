"""Context-budget adjustment and narrowly scoped migration for saved Reddit runs."""

BUDGET = '''    requested_max_new_tokens = max_new_tokens
    context_limit = getattr(model.config, "max_position_embeddings", None)
    available = context_limit - int(input_ids.shape[-1]) if context_limit else max_new_tokens
    max_new_tokens = min(max_new_tokens, available)
    if max_new_tokens <= 0:
        return {"text": "", "input_tokens": int(input_ids.shape[-1]),
                "generated_tokens": 0, "elapsed_seconds": 0.0,
                "hit_token_limit": False, "context_capacity_exceeded": True,
                "requested_max_new_tokens": requested_max_new_tokens,
                "effective_max_new_tokens": 0, "context_budget_adjusted": True}
'''

MIGRATION = '''
manifest["context_policy"] = "preserve-input-cap-generation-to-remaining-v1"
if manifest_path.exists():
    previous = json.loads(manifest_path.read_text())
    candidate = dict(manifest)
    candidate.pop("context_policy", None)
    candidate["implementation_sha256"] = LEGACY_IMPLEMENTATION_SHA256
    if previous == candidate:
        # Only completed rows accepted under the former full-budget guard may resume.
        for method in RUN_METHODS:
            saved = load_results(METHOD_CONFIG[method]["path"])
            if not saved.empty:
                if not saved["routed_id_hash"].eq(routed_id_hash).all():
                    raise ValueError("Cannot migrate mismatched input IDs.")
                for raw in saved["stage_logs_json"]:
                    stages = json.loads(raw)
                    expected = 1 if method == "direct" else 4
                    if len(stages) != expected or any(
                        int(s["input_tokens"]) + LLAMA2_MAX_NEW_TOKENS > 4096
                        for s in stages
                    ):
                        raise ValueError("Legacy row cannot be reused under the context policy.")
        backup = manifest_path.with_name("experiment_manifest.before_context_fix.json")
        if not backup.exists():
            backup.write_text(json.dumps(previous, indent=2))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("Verified completed rows; preserving them under the context-budget fix.")
'''


def patch_helpers(text):
    start = text.index('    context_limit = getattr(model.config, "max_position_embeddings", None)')
    end = text.index('    terminators = ', start)
    text = text[:start] + BUDGET + text[end:]
    marker = '        "hit_token_limit": bool(generated.shape[-1] >= max_new_tokens),'
    assert text.count(marker) == 1
    return text.replace(marker, marker + '''
        "requested_max_new_tokens": requested_max_new_tokens,
        "effective_max_new_tokens": int(max_new_tokens),
        "context_budget_adjusted": bool(max_new_tokens < requested_max_new_tokens),
        "context_capacity_exceeded": False,''')
