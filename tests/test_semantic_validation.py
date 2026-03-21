import copy
import math
import unittest

from inferoscope.validation.semantic import (
    ValidationIssue,
    validate_layout_semantics,
    validate_raw_event_semantics,
    validate_run_bundle_semantics,
)


def make_valid_raw_event() -> dict:
    return {
        "event_type": "token_complete",
        "schema_version": "raw/v0.1.0",
        "run_id": "run-001",
        "token_index": 0,
        "token_id": 42,
        "token_text": "hello",
        "context_length": 5,
        "timing_ms": {
            "decode_start": 10.0,
            "decode_end": 25.0,
            "decode_duration": 15.0,
        },
        "layers": [
            {
                "layer_index": 0,
                "layer_kind": "moe",
                "num_total_experts": 4,
                "num_active_experts": 2,
                "router_probs": [0.6, 0.3, 0.1, 0.0],
                "topk_indices": [0, 1],
                "topk_probs": [0.6, 0.3],
                "entropy": 0.8979457248567797,
                "normalized_entropy": 0.647730922119161,
                "top1_prob": 0.6,
                "top1_top2_margin": 0.3,
            }
        ],
    }


def entropy_for(router_probs: list[float]) -> float:
    return -sum(prob * math.log(prob) for prob in router_probs if prob > 0.0)


def make_valid_manifest() -> dict:
    return {
        "schema_version": "manifest/v0.1.0",
        "run_id": "run-001",
        "created_at": "2026-03-18T12:00:00Z",
        "model_id": "allenai/OLMoE-1B-7B-0125",
        "tokenizer_id": "allenai/OLMoE-1B-7B-0125",
        "prompt": {"text": "hello"},
        "seed": 7,
        "generation_config": {
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "capture_config": {
            "trace_level": "full-router-probs",
            "device": "mps",
        },
        "artifact_versions": {
            "raw_event_schema_version": "raw/v0.1.0",
            "derived_event_schema_version": "derived/v0.1.0-provisional",
            "layout_schema_version": "layout/v0.1.0",
            "motif_ledger_schema_version": "motif_ledger/v0.1.0-provisional",
            "contingency_schema_version": "contingency/v0.1.0-provisional",
        },
        "derivation_version": "motifs/v0.1.0-alpha",
        "derivation_config_id": "motifs/default-alpha",
    }


def make_valid_layout() -> dict:
    return {
        "schema_version": "layout/v0.1.0",
        "run_id": "run-001",
        "layout_id": "default-grid",
        "layout_kind": "layer-grid",
        "layers": [
            {
                "layer_index": 0,
                "positions": [
                    {"expert_index": 0, "x": 0.0, "y": 0.0, "z": 0.0},
                    {"expert_index": 1, "x": 1.0, "y": 0.0, "z": 0.0},
                    {"expert_index": 2, "x": 0.0, "y": 1.0, "z": 0.0},
                    {"expert_index": 3, "x": 1.0, "y": 1.0, "z": 0.0},
                ],
            }
        ],
    }


def make_valid_derived_event() -> dict:
    return {
        "event_type": "token_derived",
        "schema_version": "derived/v0.1.0-provisional",
        "run_id": "run-001",
        "token_index": 0,
        "derivation_version": "motifs/v0.1.0-alpha",
        "derivation_config_id": "motifs/default-alpha",
        "derived_payload": {},
    }


def make_valid_motif_ledger() -> dict:
    return {
        "schema_version": "motif_ledger/v0.1.0-provisional",
        "run_id": "run-001",
        "derivation_version": "motifs/v0.1.0-alpha",
        "derivation_config_id": "motifs/default-alpha",
        "ledger_payload": {},
    }


def make_valid_contingency() -> dict:
    return {
        "schema_version": "contingency/v0.1.0-provisional",
        "run_id": "run-001",
        "derivation_version": "motifs/v0.1.0-alpha",
        "derivation_config_id": "motifs/default-alpha",
        "contingency_payload": {},
    }


class SemanticValidationApiTests(unittest.TestCase):
    def test_validation_issue_dataclass_is_constructible(self) -> None:
        issue = ValidationIssue(
            scope="raw_event",
            code="example",
            message="example message",
        )

        self.assertEqual(issue.scope, "raw_event")
        self.assertEqual(issue.code, "example")
        self.assertEqual(issue.message, "example message")


class RawEventSemanticValidationTests(unittest.TestCase):
    def test_validate_raw_event_semantics_accepts_valid_event(self) -> None:
        issues = validate_raw_event_semantics(make_valid_raw_event())
        self.assertEqual(issues, [])

    def test_validate_raw_event_semantics_reports_router_probs_length_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["router_probs"] = [0.6, 0.3, 0.1]

        issues = validate_raw_event_semantics(event)

        self.assertIn("router_probs_length_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_router_probs_sum_mismatch(self) -> None:
        event = make_valid_raw_event()
        layer = event["layers"][0]
        layer["router_probs"] = [0.5, 0.3, 0.1, 0.0]
        layer["topk_probs"] = [0.5, 0.3]
        layer["top1_prob"] = 0.5
        layer["top1_top2_margin"] = 0.2
        layer["entropy"] = entropy_for(layer["router_probs"])
        layer["normalized_entropy"] = layer["entropy"] / math.log(layer["num_total_experts"])

        issues = validate_raw_event_semantics(event)

        self.assertIn("router_probs_sum_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_topk_index_out_of_bounds(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["topk_indices"] = [0, 5]

        issues = validate_raw_event_semantics(event)

        self.assertIn("topk_index_out_of_bounds", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_topk_prob_mismatch(self) -> None:
        event = make_valid_raw_event()
        layer = event["layers"][0]
        layer["topk_probs"] = [0.6, 0.25]
        layer["top1_top2_margin"] = 0.35

        issues = validate_raw_event_semantics(event)

        self.assertIn("topk_prob_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_decode_duration_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["timing_ms"]["decode_duration"] = 14.0

        issues = validate_raw_event_semantics(event)

        self.assertIn("decode_duration_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_top1_prob_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["top1_prob"] = 0.55

        issues = validate_raw_event_semantics(event)

        self.assertIn("top1_prob_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_top1_top2_margin_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["top1_top2_margin"] = 0.25

        issues = validate_raw_event_semantics(event)

        self.assertIn("top1_top2_margin_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_num_active_experts_exceeds_total(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["num_active_experts"] = 5
        # This fixture is intentionally invalid in more than one way; we are
        # specifically pinning the dedicated count-overflow diagnostic.
        event["layers"][0]["topk_indices"] = [0, 1, 2, 3, 3]
        event["layers"][0]["topk_probs"] = [0.6, 0.3, 0.1, 0.0, 0.0]

        issues = validate_raw_event_semantics(event)

        self.assertIn(
            "num_active_experts_exceeds_num_total_experts",
            {issue.code for issue in issues},
        )

    def test_validate_raw_event_semantics_reports_duplicate_topk_index(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["num_active_experts"] = 4
        event["layers"][0]["topk_indices"] = [0, 1, 1, 2]
        event["layers"][0]["topk_probs"] = [0.6, 0.3, 0.3, 0.1]

        issues = validate_raw_event_semantics(event)

        self.assertIn("duplicate_topk_index", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_duplicate_layer_index(self) -> None:
        event = make_valid_raw_event()
        duplicate_layer = copy.deepcopy(event["layers"][0])
        duplicate_layer["top1_prob"] = 0.55
        event["layers"].append(duplicate_layer)

        issues = validate_raw_event_semantics(event)

        self.assertIn("duplicate_layer_index", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_topk_selection_mismatch(self) -> None:
        event = make_valid_raw_event()
        layer = event["layers"][0]
        layer["topk_indices"] = [1, 2]
        layer["topk_probs"] = [0.3, 0.1]
        layer["top1_prob"] = 0.6
        layer["top1_top2_margin"] = 0.2

        issues = validate_raw_event_semantics(event)

        self.assertIn("topk_selection_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_non_positive_num_active_experts(self) -> None:
        event = make_valid_raw_event()
        layer = event["layers"][0]
        layer["num_active_experts"] = 0
        layer["topk_indices"] = []
        layer["topk_probs"] = []
        layer["top1_prob"] = 0.6
        layer["top1_top2_margin"] = 0.0

        issues = validate_raw_event_semantics(event)

        self.assertIn("num_active_experts_must_be_positive", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_entropy_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["entropy"] = 0.8

        issues = validate_raw_event_semantics(event)

        self.assertIn("entropy_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_reports_normalized_entropy_mismatch(self) -> None:
        event = make_valid_raw_event()
        event["layers"][0]["normalized_entropy"] = 0.6

        issues = validate_raw_event_semantics(event)

        self.assertIn("normalized_entropy_mismatch", {issue.code for issue in issues})

    def test_validate_raw_event_semantics_accepts_tied_topk_in_any_order(self) -> None:
        event = make_valid_raw_event()
        layer = event["layers"][0]
        layer["router_probs"] = [0.5, 0.5, 0.0, 0.0]
        layer["topk_indices"] = [1, 0]
        layer["topk_probs"] = [0.5, 0.5]
        layer["top1_prob"] = 0.5
        layer["top1_top2_margin"] = 0.0
        layer["entropy"] = math.log(2.0)
        layer["normalized_entropy"] = 0.5

        issues = validate_raw_event_semantics(event)

        self.assertEqual(issues, [])

class LayoutSemanticValidationTests(unittest.TestCase):
    def test_validate_layout_semantics_reports_duplicate_expert_index(self) -> None:
        layout = make_valid_layout()
        layout["layers"][0]["positions"][1]["expert_index"] = 0

        issues = validate_layout_semantics(layout)

        self.assertIn("duplicate_expert_index", {issue.code for issue in issues})


class RunBundleSemanticValidationTests(unittest.TestCase):
    def test_validate_run_bundle_semantics_reports_raw_event_schema_version_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        raw_events[0]["schema_version"] = "raw/v9.9.9"
        layout = make_valid_layout()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("raw_event_schema_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_uses_manifest_expected_raw_event_schema_version(self) -> None:
        manifest = make_valid_manifest()
        manifest["artifact_versions"]["raw_event_schema_version"] = "raw/v9.9.9"
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("raw_event_schema_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_layout_schema_version_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        layout["schema_version"] = "layout/v9.9.9"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("layout_schema_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_derived_event_schema_version_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event()]
        derived_events[0]["schema_version"] = "derived/v9.9.9"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
        )

        self.assertIn("derived_event_schema_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_motif_ledger_schema_version_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        motif_ledger = make_valid_motif_ledger()
        motif_ledger["schema_version"] = "motif_ledger/v9.9.9"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            motif_ledger=motif_ledger,
        )

        self.assertIn("motif_ledger_schema_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_contingency_schema_version_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        contingency = make_valid_contingency()
        contingency["schema_version"] = "contingency/v9.9.9"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            contingency=contingency,
        )

        self.assertIn("contingency_schema_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_raw_event_run_id_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        raw_events[0]["run_id"] = "other-run"
        layout = make_valid_layout()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("run_id_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_layout_run_id_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        layout["run_id"] = "other-run"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("run_id_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_derived_event_run_id_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event()]
        derived_events[0]["run_id"] = "other-run"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
        )

        self.assertIn("run_id_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_support_artifact_run_id_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        motif_ledger = make_valid_motif_ledger()
        motif_ledger["run_id"] = "other-run"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            motif_ledger=motif_ledger,
            contingency=make_valid_contingency(),
        )

        self.assertIn("run_id_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_layout_expert_count_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        layout["layers"][0]["positions"].pop()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("layout_expert_count_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_raw_layer_expert_count_inconsistent(self) -> None:
        manifest = make_valid_manifest()
        first = make_valid_raw_event()
        second = copy.deepcopy(make_valid_raw_event())
        second["token_index"] = 1
        second["token_id"] = 43
        second["token_text"] = "world"
        second_layer = second["layers"][0]
        second_layer["num_total_experts"] = 5
        second_layer["router_probs"] = [0.55, 0.25, 0.1, 0.05, 0.05]
        second_layer["topk_indices"] = [0, 1]
        second_layer["topk_probs"] = [0.55, 0.25]
        second_layer["top1_prob"] = 0.55
        second_layer["top1_top2_margin"] = 0.3
        second_layer["entropy"] = entropy_for(second_layer["router_probs"])
        second_layer["normalized_entropy"] = second_layer["entropy"] / math.log(
            second_layer["num_total_experts"]
        )
        raw_events = [first, second]

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            make_valid_layout(),
        )

        self.assertIn("raw_layer_expert_count_inconsistent", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_duplicate_raw_token_index(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event(), copy.deepcopy(make_valid_raw_event())]
        raw_events[1]["token_id"] = 43
        raw_events[1]["token_text"] = "world"
        layout = make_valid_layout()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("duplicate_raw_token_index", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_non_monotonic_raw_token_sequence(self) -> None:
        manifest = make_valid_manifest()
        first = make_valid_raw_event()
        first["token_index"] = 1
        second = copy.deepcopy(make_valid_raw_event())
        second["token_index"] = 0
        second["token_id"] = 43
        second["token_text"] = "world"
        raw_events = [first, second]
        layout = make_valid_layout()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("raw_token_index_sequence_invalid", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_gapped_raw_token_sequence(self) -> None:
        manifest = make_valid_manifest()
        first = make_valid_raw_event()
        second = copy.deepcopy(make_valid_raw_event())
        second["token_index"] = 2
        second["token_id"] = 43
        second["token_text"] = "world"
        raw_events = [first, second]
        layout = make_valid_layout()

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("raw_token_index_sequence_invalid", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_requires_matching_derivation_metadata_across_bundle(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event()]
        derived_events[0]["derivation_config_id"] = "motifs/other-alpha"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
            motif_ledger=make_valid_motif_ledger(),
            contingency=make_valid_contingency(),
        )

        self.assertIn("derivation_config_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_derivation_version_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event()]
        derived_events[0]["derivation_version"] = "motifs/other-alpha"

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
        )

        self.assertIn("derivation_version_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_orphan_derived_token_index(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event()]
        derived_events[0]["token_index"] = 99

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
        )

        self.assertIn("derived_token_index_missing_from_raw", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_duplicate_derived_token_index(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        derived_events = [make_valid_derived_event(), copy.deepcopy(make_valid_derived_event())]

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
            derived_events=copy.deepcopy(derived_events),
        )

        self.assertIn("duplicate_derived_token_index", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_reports_layout_expert_index_set_mismatch(self) -> None:
        manifest = make_valid_manifest()
        raw_events = [make_valid_raw_event()]
        layout = make_valid_layout()
        layout["layers"][0]["positions"][3]["expert_index"] = 99

        issues = validate_run_bundle_semantics(
            manifest,
            raw_events,
            layout,
        )

        self.assertIn("layout_expert_index_set_mismatch", {issue.code for issue in issues})

    def test_validate_run_bundle_semantics_accepts_valid_bundle_with_optional_artifacts(self) -> None:
        issues = validate_run_bundle_semantics(
            make_valid_manifest(),
            [make_valid_raw_event()],
            make_valid_layout(),
            derived_events=[make_valid_derived_event()],
            motif_ledger=make_valid_motif_ledger(),
            contingency=make_valid_contingency(),
        )

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
