from scripts import decision_dominance_audit as dda


def test_decision_dominance_audit_smoke(tmp_path):
    out_dir = tmp_path / "out"
    result = dda.run_audit(episodes=1, steps=10, seed=0, out_dir=str(out_dir))
    md_path = result["md"]
    csv_path = result["csv"]

    assert out_dir.exists()
    assert md_path.endswith("decision_dominance_audit.md")
    assert csv_path.endswith("decision_dominance_audit.csv")

    md_text = (out_dir / "decision_dominance_audit.md").read_text(encoding="utf-8")
    assert "Argmin Distribution" in md_text
    assert "V2V vs RSU Delta" in md_text
    assert "Top V2V Worst Samples" in md_text
    assert "Confusion Matrix" in md_text
