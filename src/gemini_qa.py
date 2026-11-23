"""Gemini QA validator for the French QCM pipeline."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BATCH_SIZE = 5
MAX_CONTEXT_CHARS = 3500
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "answerable": {"type": "boolean"},
                    "predicted_index": {"type": "integer"},
                    "supported_option_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "evidence": {"type": "string"},
                },
                "required": [
                    "question_id",
                    "answerable",
                    "predicted_index",
                    "supported_option_indices",
                    "evidence",
                ],
            },
        }
    },
    "required": ["evaluations"],
}


def create_gemini_client(api_key: str | None = None) -> genai.Client:
    """Create a Gemini client using a key from the project .env or environment."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Clé Gemini absente. Renseignez GEMINI_API_KEY dans le fichier .env "
            "à la racine du projet, ou définissez cette variable d'environnement."
        )
    return genai.Client(api_key=key)


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _generate_json(client: Any, prompt: str) -> dict[str, Any]:
    response = client.interactions.create(
        model=_model_name(),
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _VALIDATION_SCHEMA,
        },
        store=False,
    )
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Gemini a renvoyé une réponse JSON vide.")
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini n'a pas renvoyé un JSON valide.") from exc
    if not isinstance(result, dict):
        raise ValueError("La réponse Gemini doit être un objet JSON.")
    return result


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def _options_for_row(row: Any) -> list[str]:
    raw_options = row.options
    options = json.loads(raw_options) if isinstance(raw_options, str) else raw_options
    if (
        not isinstance(options, list)
        or len(options) != 4
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError(f"Le QCM {row.question_id} doit avoir quatre options non vides.")
    return options


def _context_for_row(row: Any, passages: pd.DataFrame) -> str:
    document = passages.loc[passages["source_doc"] == row.source_doc]
    page = int(row.source_page)
    nearby = document.loc[document["page"].between(page - 1, page + 1)]
    if nearby.empty:
        nearby = document.loc[document["page"] == page]
    evidence = str(row.evidence).strip()
    context_parts = [evidence]
    context_parts.extend(
        str(text).strip()
        for text in nearby["text"].astype(str)
        if _normalize(str(text)) != _normalize(evidence)
    )
    context = "\n".join(part for part in context_parts if part)
    if not context:
        raise ValueError(f"Aucun contexte pédagogique disponible pour {row.question_id}.")
    return context[:MAX_CONTEXT_CHARS]


def _prompt_for_batch(items: list[dict[str, Any]]) -> str:
    serialized = json.dumps(items, ensure_ascii=False, indent=2)
    return (
        "Tu es un validateur indépendant de QCM en français. Évalue chaque question "
        "uniquement à partir de son contexte pédagogique. Ne suppose pas qu'une option "
        "est correcte parce qu'elle semble plausible.\n"
        "Pour chaque QCM, indique si le contexte permet de répondre, l'index (0 à 3) "
        "de l'option que tu juges correcte, la liste des index de toutes les options "
        "que le contexte permet de considérer correctes, et une courte citation "
        "EXACTE du contexte qui justifie ton choix. Si aucune réponse n'est étayée, "
        "mets answerable=false, predicted_index=-1, supported_option_indices=[] et "
        "evidence=\"\". Si plusieurs options sont étayées, indique tous leurs index.\n"
        "Les index commencent à zéro. Reproduis chaque question_id exactement et "
        "retourne une évaluation par question. Réponds uniquement au format JSON "
        "demandé.\n\n"
        f"QCM à évaluer :\n{serialized}"
    )


def validate_mcqs_with_gemini(
    candidates: pd.DataFrame,
    passages: pd.DataFrame,
    client: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Validate QCMs in batches with Gemini and verify quoted evidence locally.

    The API receives the question, its four choices, and nearby course passages.
    It does not receive the reference answer index.
    """
    if batch_size < 1:
        raise ValueError("batch_size doit être supérieur ou égal à 1.")
    if candidates.empty:
        raise ValueError("Aucun QCM à valider.")
    required_candidate_columns = {
        "question_id",
        "question",
        "options",
        "correct_index",
        "source_doc",
        "source_page",
        "evidence",
    }
    missing = required_candidate_columns.difference(candidates.columns)
    if missing:
        raise KeyError(f"Colonnes candidates absentes : {sorted(missing)}")
    required_passage_columns = {"source_doc", "page", "text"}
    missing = required_passage_columns.difference(passages.columns)
    if missing:
        raise KeyError(f"Colonnes passages absentes : {sorted(missing)}")
    if candidates["question_id"].duplicated().any():
        raise ValueError("Les question_id des candidats doivent être uniques.")

    results: list[dict[str, Any]] = []
    rows = list(candidates.itertuples(index=False))
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        batch_inputs = []
        expected_rows: dict[str, Any] = {}
        contexts: dict[str, str] = {}
        for row in batch_rows:
            question_id = str(row.question_id)
            if not str(row.question).strip():
                raise ValueError(f"La question {question_id} est vide.")
            try:
                correct_index = int(row.correct_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Index de référence invalide pour {question_id}."
                ) from exc
            if correct_index not in {0, 1, 2, 3}:
                raise ValueError(f"Index de référence invalide pour {question_id}.")
            options = _options_for_row(row)
            if len({_normalize(option) for option in options}) != 4:
                results.append(
                    {
                        "question_id": question_id,
                        "qa_evidence": "",
                        "qa_model_name": _model_name(),
                        "qa_predicted_index": -1,
                        "qa_predicted_answer": "",
                        "qa_supported_option_indices": "[]",
                        "qa_answerable": False,
                        "qa_evidence_supported": False,
                        "qa_unique": False,
                        "qa_correct": False,
                        "accepted": False,
                        "rejection_reason": "options_dupliquees_avant_gemini",
                    }
                )
                continue
            context = _context_for_row(row, passages)
            batch_inputs.append(
                {
                    "question_id": question_id,
                    "question": str(row.question),
                    "options": options,
                    "context": context,
                }
            )
            expected_rows[question_id] = row
            contexts[question_id] = context

        if not batch_inputs:
            continue
        payload = _generate_json(client, _prompt_for_batch(batch_inputs))
        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            raise ValueError("La réponse Gemini ne contient pas la liste evaluations.")
        expected_ids = set(expected_rows)
        returned_ids = [
            item.get("question_id") if isinstance(item, dict) else None
            for item in evaluations
        ]
        if len(returned_ids) != len(expected_ids) or set(returned_ids) != expected_ids:
            raise ValueError(
                "Gemini doit retourner exactement une évaluation pour chaque question du lot."
            )

        for item in evaluations:
            question_id = item["question_id"]
            row = expected_rows[question_id]
            answerable = item["answerable"]
            predicted_index = item["predicted_index"]
            supported_indices = item["supported_option_indices"]
            evidence = item["evidence"]
            if not isinstance(answerable, bool):
                raise ValueError(f"answerable doit être booléen pour {question_id}.")
            if type(predicted_index) is not int or predicted_index not in {-1, 0, 1, 2, 3}:
                raise ValueError(f"Index de réponse Gemini invalide pour {question_id}.")
            if (
                not isinstance(supported_indices, list)
                or any(type(index) is not int or index not in {0, 1, 2, 3} for index in supported_indices)
                or len(set(supported_indices)) != len(supported_indices)
            ):
                raise ValueError(f"Liste d'options étayées invalide pour {question_id}.")
            if not isinstance(evidence, str):
                raise ValueError(f"La preuve doit être du texte pour {question_id}.")

            context_words = _normalize(contexts[question_id])
            evidence_words = _normalize(evidence)
            evidence_supported = bool(evidence_words) and (
                f" {evidence_words} " in f" {context_words} "
            )
            unique = len(supported_indices) == 1
            correct_index = int(row.correct_index)
            correct = (
                predicted_index == correct_index
                and supported_indices == [correct_index]
            )
            accepted = answerable and evidence_supported and unique and correct
            reasons = []
            if not answerable:
                reasons.append("question_non_etayee_par_gemini")
            if not evidence_supported:
                reasons.append("preuve_gemini_absente_du_contexte")
            if not unique:
                reasons.append("reponse_non_unique_selon_gemini")
            if not correct:
                reasons.append("option_gemini_differente_de_la_reference")
            options = _options_for_row(row)
            results.append(
                {
                    "question_id": question_id,
                    "qa_evidence": evidence,
                    "qa_model_name": _model_name(),
                    "qa_predicted_index": predicted_index,
                    "qa_predicted_answer": (
                        options[predicted_index] if predicted_index >= 0 else ""
                    ),
                    "qa_supported_option_indices": json.dumps(supported_indices),
                    "qa_answerable": answerable,
                    "qa_evidence_supported": evidence_supported,
                    "qa_unique": unique,
                    "qa_correct": correct,
                    "accepted": accepted,
                    "rejection_reason": ";".join(reasons),
                }
            )

    return candidates.merge(
        pd.DataFrame.from_records(results),
        on="question_id",
        validate="one_to_one",
    )
