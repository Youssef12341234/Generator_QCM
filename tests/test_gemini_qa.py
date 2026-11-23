import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.gemini_qa import create_gemini_client, validate_mcqs_with_gemini


class FakeInteractions:
    def __init__(self, evaluations):
        self.evaluations = evaluations
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"evaluations": self.evaluations})
        )


class FakeClient:
    def __init__(self, evaluations):
        self.interactions = FakeInteractions(evaluations)


class GeminiQAValidationTests(unittest.TestCase):
    def setUp(self):
        self.context = "La photosynthèse transforme l'énergie lumineuse en énergie chimique."
        self.candidates = pd.DataFrame(
            [
                {
                    "question_id": "Q0001",
                    "question": "Que transforme la photosynthèse ?",
                    "options": json.dumps(
                        [
                            "L'énergie lumineuse en énergie chimique",
                            "L'eau en oxygène uniquement",
                            "Le glucose en lumière",
                            "Le carbone en eau",
                        ]
                    ),
                    "correct_index": 0,
                    "source_doc": "cours.pdf",
                    "source_page": 2,
                    "evidence": self.context,
                }
            ]
        )
        self.passages = pd.DataFrame(
            [
                {
                    "source_doc": "cours.pdf",
                    "page": 2,
                    "text": self.context,
                }
            ]
        )

    def test_accepts_only_correct_unique_answer_with_verbatim_evidence(self):
        client = FakeClient(
            [
                {
                    "question_id": "Q0001",
                    "answerable": True,
                    "predicted_index": 0,
                    "supported_option_indices": [0],
                    "evidence": self.context,
                }
            ]
        )

        result = validate_mcqs_with_gemini(
            self.candidates, self.passages, client
        )

        self.assertTrue(bool(result.loc[0, "accepted"]))
        self.assertTrue(bool(result.loc[0, "qa_evidence_supported"]))
        call = client.interactions.calls[0]
        self.assertFalse(call["store"])
        self.assertNotIn("correct_index", call["input"])

    def test_rejects_answer_with_unverifiable_evidence(self):
        client = FakeClient(
            [
                {
                    "question_id": "Q0001",
                    "answerable": True,
                    "predicted_index": 0,
                    "supported_option_indices": [0],
                    "evidence": "La réponse est certainement la première.",
                }
            ]
        )

        result = validate_mcqs_with_gemini(
            self.candidates, self.passages, client
        )

        self.assertFalse(bool(result.loc[0, "accepted"]))
        self.assertFalse(bool(result.loc[0, "qa_evidence_supported"]))

    def test_requires_exactly_one_response_for_each_qcm(self):
        client = FakeClient([])

        with self.assertRaisesRegex(ValueError, "exactement une évaluation"):
            validate_mcqs_with_gemini(
                self.candidates, self.passages, client
            )

    def test_rejects_duplicate_choices_without_calling_the_api(self):
        candidates = self.candidates.copy()
        options = json.loads(candidates.loc[0, "options"])
        options[1] = options[0]
        candidates.loc[0, "options"] = json.dumps(options)
        client = FakeClient([])

        result = validate_mcqs_with_gemini(candidates, self.passages, client)

        self.assertFalse(bool(result.loc[0, "accepted"]))
        self.assertEqual(
            result.loc[0, "rejection_reason"],
            "options_dupliquees_avant_gemini",
        )
        self.assertEqual(client.interactions.calls, [])

    def test_api_key_is_required_from_environment_or_argument(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                create_gemini_client()


if __name__ == "__main__":
    unittest.main()
