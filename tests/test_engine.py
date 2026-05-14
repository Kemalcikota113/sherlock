# End-to-end smoke test against the real Google API.
# How to use: set RUN_INTEGRATION=1 and provide a valid GOOGLE_API_KEY.

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run the real-API smoke test.",
)


def test_grounded_answer_and_refusal(tmp_path):
    from app.engine import SherlockEngine

    engine = SherlockEngine(persist_directory=str(tmp_path / "chroma"))

    case = tmp_path / "case.txt"
    case.write_text(
        "Case File #101: The Manor Mystery.\n"
        "Witness: Mrs. Hudson.\n"
        "Alibi: Mrs. Hudson was in the kitchen making tea at midnight.\n"
        "Suspect: The Butler was seen in the garden with a silver tray."
    )

    engine.ingest(str(case), "case.txt")

    answer = engine.query("What was Mrs. Hudson's alibi?")["answer"].lower()
    assert "tea" in answer or "kitchen" in answer

    refusal = engine.query("What is the suspect's middle name?")["answer"].lower()
    assert "enough evidence" in refusal
