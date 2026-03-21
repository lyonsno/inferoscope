import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_SCHEMA_DIR = REPO_ROOT / "schema" / "v0.1.0"
PACKAGED_SCHEMA_DIR = REPO_ROOT / "inferoscope" / "validation" / "schemas" / "v0.1.0"


def load_json(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text())


class SchemaContractAlignmentTests(unittest.TestCase):
    def test_raw_trace_schema_has_stable_top_level_contract(self) -> None:
        schema = load_json("schema/v0.1.0/raw_trace_event.schema.json")

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["event_type"]["const"], "token_complete")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "raw/v0.1.0")
        self.assertEqual(
            schema["required"],
            [
                "event_type",
                "schema_version",
                "run_id",
                "token_index",
                "token_id",
                "token_text",
                "context_length",
                "timing_ms",
                "layers",
            ],
        )

    def test_raw_trace_schema_only_advertises_moe_layers_for_v0_1_0(self) -> None:
        schema = load_json("schema/v0.1.0/raw_trace_event.schema.json")

        layer_kind_enum = schema["$defs"]["layer"]["properties"]["layer_kind"]["enum"]

        self.assertEqual(layer_kind_enum, ["moe"])

    def test_raw_trace_schema_requires_positive_num_active_experts(self) -> None:
        canonical_schema = load_json("schema/v0.1.0/raw_trace_event.schema.json")
        packaged_schema = load_json("inferoscope/validation/schemas/v0.1.0/raw_trace_event.schema.json")

        canonical_minimum = canonical_schema["$defs"]["layer"]["properties"]["num_active_experts"]["minimum"]
        packaged_minimum = packaged_schema["$defs"]["layer"]["properties"]["num_active_experts"]["minimum"]

        self.assertEqual(canonical_minimum, 1)
        self.assertEqual(packaged_minimum, 1)

    def test_packaged_runtime_schemas_match_canonical_schema_directory(self) -> None:
        canonical_files = sorted(path.name for path in CANONICAL_SCHEMA_DIR.glob("*.json"))
        packaged_files = sorted(path.name for path in PACKAGED_SCHEMA_DIR.glob("*.json"))

        self.assertEqual(packaged_files, canonical_files)

        for filename in canonical_files:
            with self.subTest(filename=filename):
                self.assertEqual(
                    load_json(f"inferoscope/validation/schemas/v0.1.0/{filename}"),
                    load_json(f"schema/v0.1.0/{filename}"),
                )

    def test_manifest_schema_has_stable_top_level_contract(self) -> None:
        schema = load_json("schema/v0.1.0/manifest.schema.json")

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], "manifest/v0.1.0")
        self.assertIn("run_id", schema["required"])
        self.assertIn("created_at", schema["required"])
        self.assertIn("model_id", schema["required"])
        self.assertIn("tokenizer_id", schema["required"])
        self.assertIn("prompt", schema["required"])
        self.assertIn("artifact_versions", schema["required"])

    def test_manifest_schema_requires_derivation_provenance_fields(self) -> None:
        schema = load_json("schema/v0.1.0/manifest.schema.json")

        self.assertIn("derivation_version", schema["properties"])
        self.assertIn("derivation_config_id", schema["properties"])
        self.assertIn("derivation_version", schema["required"])
        self.assertIn("derivation_config_id", schema["required"])

    def test_layout_schema_has_stable_top_level_contract(self) -> None:
        schema = load_json("schema/v0.1.0/layout.schema.json")

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], "layout/v0.1.0")
        self.assertIn("run_id", schema["required"])
        self.assertIn("layout_id", schema["required"])
        self.assertIn("layers", schema["required"])

    def test_layout_schema_defines_layer_and_position_shapes(self) -> None:
        schema = load_json("schema/v0.1.0/layout.schema.json")

        layer_layout = schema["$defs"]["layer_layout"]
        expert_position = schema["$defs"]["expert_position"]

        self.assertFalse(layer_layout["additionalProperties"])
        self.assertEqual(layer_layout["required"], ["layer_index", "positions"])
        self.assertFalse(expert_position["additionalProperties"])
        self.assertEqual(expert_position["required"], ["expert_index", "x", "y", "z"])

    def test_provisional_support_schemas_require_derivation_config_id(self) -> None:
        motif_ledger = load_json("schema/v0.1.0/motif_ledger.schema.json")
        contingency = load_json("schema/v0.1.0/contingency.schema.json")

        self.assertIn("derivation_config_id", motif_ledger["properties"])
        self.assertIn("derivation_config_id", motif_ledger["required"])
        self.assertIn("derivation_config_id", contingency["properties"])
        self.assertIn("derivation_config_id", contingency["required"])

    def test_docs_explicitly_split_schema_and_semantic_validation(self) -> None:
        schema_readme = (REPO_ROOT / "schema/README.md").read_text()
        proposal = (REPO_ROOT / "docs/v0.1.0_proposal.md").read_text()

        self.assertIn("semantic validation", schema_readme.lower())
        self.assertIn("semantic validation", proposal.lower())


if __name__ == "__main__":
    unittest.main()
