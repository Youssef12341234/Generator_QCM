"""Pipeline local et reproductible pour le projet de generation de QCM N11."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_STATE = 42
MIN_PASSAGE_CHARS = 55
STOPWORDS = {
    "alors", "aucun", "aussi", "autre", "aux", "avec", "avoir", "bien", "car",
    "ceci", "cela", "celle", "celles", "celui", "cependant", "certain", "cette",
    "ceux", "chaque", "comme", "comment", "dans", "deja", "depuis", "des",
    "donc", "dont", "elle", "elles", "encore", "entre", "est", "etant", "etre",
    "eux", "fait", "font", "hors", "ici", "ils", "mais", "meme", "moins",
    "notre", "nous", "par", "parce", "pas", "peut", "plus", "pour", "pourquoi",
    "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "quoi", "sans",
    "selon", "sera", "sont", "sous", "sur", "tandis", "tous", "tout", "tres",
    "une", "vers", "votre", "vous", "affirmation", "concernant", "correcte",
    "decrit", "mieux", "cours", "support", "dit",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def _ocr_page(page: fitz.Page, ocr_engine: RapidOCR) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    result, _ = ocr_engine(image)
    if not result:
        return ""
    return " ".join(
        str(item[1]).strip()
        for item in result
        if len(item) >= 3 and float(item[2]) >= 0.35 and str(item[1]).strip()
    )


def extract_course_passages(source_dir: Path, use_ocr: bool = True) -> pd.DataFrame:
    """Extrait les pages des six supports, avec OCR local pour les pages scannees."""
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if len(pdf_paths) < 5:
        raise FileNotFoundError(
            f"Au moins cinq supports PDF sont requis dans {source_dir}; "
            f"{len(pdf_paths)} fichier(s) trouve(s)."
        )

    ocr_engine = RapidOCR() if use_ocr else None
    records: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        extracted_text = [(page.extract_text() or "").strip() for page in reader.pages]
        short_pages = sum(len(text) < MIN_PASSAGE_CHARS for text in extracted_text)
        scanned_document = bool(reader.pages) and short_pages / len(reader.pages) >= 0.6
        if use_ocr and scanned_document and ocr_engine is None:
            ocr_engine = RapidOCR()
        rendered = fitz.open(str(pdf_path)) if use_ocr and scanned_document else None
        for page_index, text in enumerate(extracted_text):
            extraction_method = "texte_pdf"
            if len(text) < MIN_PASSAGE_CHARS and rendered is not None and ocr_engine is not None:
                text = _ocr_page(rendered[page_index], ocr_engine)
                extraction_method = "ocr_local"
            elif len(text) < MIN_PASSAGE_CHARS:
                extraction_method = "texte_insuffisant"
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            records.append(
                {
                    "source_doc": pdf_path.name,
                    "page": page_index + 1,
                    "extraction_method": extraction_method,
                    "text": text,
                }
            )
        if rendered is not None:
            rendered.close()

    if not records:
        raise ValueError("Aucun texte n'a pu etre extrait des supports fournis.")
    return pd.DataFrame.from_records(records)


def split_passages(pages: pd.DataFrame) -> pd.DataFrame:
    """Transforme les pages en phrases informatives, en conservant leur provenance."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in pages.itertuples(index=False):
        fragments = re.split(r"(?<=[.!?])\s+|(?<=:)\s+", row.text)
        for fragment in fragments:
            sentence = re.sub(r"\s+", " ", fragment).strip(" \t\r\n-•")
            normalized = _normalize(sentence)
            if len(sentence) < MIN_PASSAGE_CHARS or len(normalized.split()) < 8:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            rows.append(
                {
                    "passage_id": f"{row.source_doc}#p{row.page:03d}-{len(rows) + 1:05d}",
                    "source_doc": row.source_doc,
                    "page": row.page,
                    "extraction_method": row.extraction_method,
                    "text": sentence,
                }
            )
    passages = pd.DataFrame.from_records(rows)
    if passages.empty:
        raise ValueError("Aucune phrase exploitable n'a ete extraite des supports.")
    return passages


def _keywords(text: str, vectorizer: TfidfVectorizer, matrix: Any, row_index: int) -> list[str]:
    terms = vectorizer.get_feature_names_out()
    weights = matrix[row_index].toarray().ravel()
    ranked = np.argsort(weights)[::-1]
    keywords: list[str] = []
    for index in ranked:
        term = terms[index]
        if not weights[index]:
            break
        if len(term) < 4 or any(token in STOPWORDS for token in term.split()):
            continue
        if term not in keywords:
            keywords.append(term)
        if len(keywords) == 8:
            break
    return keywords


def _token_f1(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]{3,}", _normalize(left)))
    right_tokens = set(re.findall(r"[a-z0-9]{3,}", _normalize(right)))
    left_tokens -= STOPWORDS
    right_tokens -= STOPWORDS
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class LocalExtractiveQAModel:
    """Modele QA local par recuperation TF-IDF et extraction d'une phrase."""

    def __call__(self, question: str, context: str) -> dict[str, Any]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", context)
            if len(sentence.strip()) >= MIN_PASSAGE_CHARS
        ]
        if not sentences:
            return {"answer": "", "score": 0.0}
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9-]{2,}\b",
            stop_words=list(STOPWORDS),
        )
        matrix = vectorizer.fit_transform([question] + sentences)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        best_index = int(scores.argmax())
        question_terms = set(
            re.findall(r"[a-z0-9]{3,}", _normalize(question))
        ) - STOPWORDS
        answer_terms = set(
            re.findall(r"[a-z0-9]{3,}", _normalize(sentences[best_index]))
        ) - STOPWORDS
        coverage = len(question_terms & answer_terms) / max(len(question_terms), 1)
        return {
            "answer": sentences[best_index],
            "score": float(scores[best_index] * coverage),
        }


def load_qa_model(
    backend: str = "local",
    model_name: str = "illuin/camembert-base-fquad",
):
    """Charge le QA lexical local ou, sur demande, le modele neuronal CamemBERT."""
    if backend == "local":
        return LocalExtractiveQAModel()
    if backend != "camembert":
        raise ValueError("backend doit etre 'local' ou 'camembert'.")
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Le backend CamemBERT requiert transformers et torch. "
            "Installez requirements-camembert.txt."
        ) from exc
    return pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=-1,
        token=False,
    )


def generate_mcqs(passages: pd.DataFrame, limit: int = 120) -> pd.DataFrame:
    """Genere des QCM extractifs; les distracteurs proviennent des memes supports."""
    if len(passages) < 5:
        raise ValueError("Il faut au moins cinq passages pour construire les QCM.")
    texts = passages["text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9-]{2,}\b",
        stop_words=list(STOPWORDS),
        max_features=12000,
    )
    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix)
    candidates: list[dict[str, Any]] = []
    for i, passage in enumerate(passages.itertuples(index=False)):
        answer = str(passage.text).strip()
        if not MIN_PASSAGE_CHARS <= len(answer) <= 500:
            continue
        keywords = _keywords(answer, vectorizer, matrix, i)
        if not keywords:
            continue
        topic = keywords[0]
        distractor_indices = [
            j
            for j in np.argsort(similarities[i])[::-1]
            if j != i
            and passages.iloc[j]["source_doc"] == passage.source_doc
            and _normalize(topic) not in _normalize(passages.iloc[j]["text"])
            and MIN_PASSAGE_CHARS <= len(str(passages.iloc[j]["text"])) <= 500
            and _token_f1(answer, str(passages.iloc[j]["text"])) < 0.72
        ]
        if len(distractor_indices) < 3:
            distractor_indices = [
                j
                for j in np.argsort(similarities[i])[::-1]
                if j != i
                and passages.iloc[j]["source_doc"] != passage.source_doc
                and _normalize(topic) not in _normalize(passages.iloc[j]["text"])
                and MIN_PASSAGE_CHARS <= len(str(passages.iloc[j]["text"])) <= 500
                and _token_f1(answer, str(passages.iloc[j]["text"])) < 0.72
            ]
        if len(distractor_indices) < 3:
            continue
        options = [answer] + [
            str(passages.iloc[j]["text"]) for j in distractor_indices[:3]
        ]
        rng = np.random.default_rng(RANDOM_STATE + i)
        order = rng.permutation(len(options))
        shuffled = [options[index] for index in order]
        correct_index = int(np.flatnonzero(order == 0)[0])
        candidates.append(
            {
                "question_id": f"Q{i + 1:04d}",
                "source_doc": passage.source_doc,
                "source_page": int(passage.page),
                "evidence": answer,
                "topic": topic,
                "question": (
                    f"Selon le support, quelle affirmation concernant « {topic} » est correcte ?"
                ),
                "options": json.dumps(shuffled, ensure_ascii=False),
                "answer": answer,
                "correct_index": correct_index,
                "generation_method": "extractif_tfidf_template",
            }
        )
        if len(candidates) >= limit:
            break
    if not candidates:
        raise ValueError(
            "Aucun QCM n'a pu etre genere. Verifiez la qualite du texte extrait "
            "et la presence de suffisamment de passages par support."
        )
    return pd.DataFrame.from_records(candidates)


def validate_mcqs(
    candidates: pd.DataFrame,
    passages: pd.DataFrame,
    support_threshold: float = 0.55,
    uniqueness_margin: float = 0.08,
) -> pd.DataFrame:
    """Valide par QA extractif local (TF-IDF + comparaison de spans/choix)."""
    results: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        options = json.loads(row.options) if isinstance(row.options, str) else row.options
        source_passages = passages[passages["source_doc"] == row.source_doc]
        if source_passages.empty:
            evidence = ""
        else:
            evidence_matrix = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9-]{2,}\b",
                stop_words=list(STOPWORDS),
            ).fit_transform(
                [str(row.question)] + source_passages["text"].astype(str).tolist()
            )
            query_scores = cosine_similarity(evidence_matrix[0:1], evidence_matrix[1:]).ravel()
            evidence = str(source_passages.iloc[int(query_scores.argmax())]["text"])
        option_scores = [_token_f1(option, evidence) for option in options]
        predicted_index = int(np.argmax(option_scores)) if option_scores else -1
        sorted_scores = sorted(option_scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        query_support = _token_f1(str(row.question), evidence)
        answerable = bool(evidence) and query_support >= 0.05
        unique = bool(sorted_scores) and sorted_scores[0] >= support_threshold and margin >= uniqueness_margin
        correct = predicted_index == int(row.correct_index)
        accepted = answerable and unique and correct
        reasons = []
        if not answerable:
            reasons.append("question_non_etayee")
        if not unique:
            reasons.append("reponse_non_unique_ou_faiblement_etayee")
        if not correct:
            reasons.append("mauvaise_option_selectionnee_par_le_modele_QA")
        results.append(
            {
                "question_id": row.question_id,
                "qa_evidence": evidence,
                "qa_predicted_index": predicted_index,
                "qa_predicted_answer": options[predicted_index] if predicted_index >= 0 else "",
                "qa_support_score": max(option_scores, default=0.0),
                "qa_margin": margin,
                "qa_answerable": answerable,
                "qa_unique": unique,
                "qa_correct": correct,
                "accepted": accepted,
                "rejection_reason": ";".join(reasons),
            }
        )
    return candidates.merge(pd.DataFrame(results), on="question_id", validate="one_to_one")


def validate_mcqs_with_qa(
    candidates: pd.DataFrame,
    passages: pd.DataFrame,
    qa_model: Any,
    min_model_confidence: float = 0.05,
    min_option_match: float = 0.18,
    uniqueness_margin: float = 0.04,
) -> pd.DataFrame:
    """Verifie question et options a l'aide du modele QA extractif fourni."""
    results: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        options = json.loads(row.options) if isinstance(row.options, str) else list(row.options)
        document_passages = passages[
            (passages["source_doc"] == row.source_doc)
            & (passages["page"].between(int(row.source_page) - 1, int(row.source_page) + 1))
        ]
        if document_passages.empty:
            document_passages = passages[
                (passages["source_doc"] == row.source_doc)
                & (passages["page"] == int(row.source_page))
            ]
        surrounding_text = [
            str(text)
            for text in document_passages["text"].astype(str).tolist()
            if _normalize(str(text)) != _normalize(str(row.evidence))
        ]
        context = " ".join([str(row.evidence), *surrounding_text])[:2800]
        if not context:
            answer, confidence = "", 0.0
        else:
            prediction = qa_model(question=str(row.question), context=context)
            answer = str(prediction.get("answer", "")).strip()
            confidence = float(prediction.get("score", 0.0))
        scores = [_token_f1(option, answer) for option in options]
        predicted_index = int(np.argmax(scores)) if scores else -1
        ordered_scores = sorted(scores, reverse=True)
        margin = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else 0.0
        answerable = bool(answer) and confidence >= min_model_confidence
        unique = bool(scores) and scores[predicted_index] >= min_option_match and margin >= uniqueness_margin
        correct = predicted_index == int(row.correct_index)
        accepted = answerable and unique and correct
        reasons = []
        if not answerable:
            reasons.append("question_non_etayee_par_le_modele_QA")
        if not unique:
            reasons.append("options_non_uniques_selon_le_modele_QA")
        if not correct:
            reasons.append("option_QA_differente_de_la_reponse_reference")
        results.append(
            {
                "question_id": row.question_id,
                "qa_evidence": answer,
                "qa_model_confidence": confidence,
                "qa_predicted_index": predicted_index,
                "qa_predicted_answer": options[predicted_index] if predicted_index >= 0 else "",
                "qa_option_match": max(scores, default=0.0),
                "qa_margin": margin,
                "qa_answerable": answerable,
                "qa_unique": unique,
                "qa_correct": correct,
                "accepted": accepted,
                "rejection_reason": ";".join(reasons),
            }
        )
    return candidates.merge(pd.DataFrame(results), on="question_id", validate="one_to_one")


def make_filter_challenges(candidates: pd.DataFrame, count: int = 12) -> pd.DataFrame:
    """Construit des controles de robustesse: ambiguite et question non etayee."""
    challenges: list[dict[str, Any]] = []
    sample = candidates.head(count).copy()
    for row in sample.itertuples(index=False):
        options = json.loads(row.options) if isinstance(row.options, str) else list(row.options)
        ambiguous_options = list(options)
        ambiguous_options[(int(row.correct_index) + 1) % len(ambiguous_options)] = options[
            int(row.correct_index)
        ]
        challenges.append(
            {
                **row._asdict(),
                "question_id": f"{row.question_id}_AMB",
                "options": json.dumps(ambiguous_options, ensure_ascii=False),
                "challenge_type": "ambiguite_introduite",
            }
        )
        unsupported_options = [
            "Une reponse a confirmer par le support original."
            for _ in options
        ]
        challenges.append(
            {
                **row._asdict(),
                "question_id": f"{row.question_id}_ABS",
                "topic": "notion absente du support",
                "question": "Quelle affirmation le support donne-t-il sur une notion absente ?",
                "options": json.dumps(unsupported_options, ensure_ascii=False),
                "answer": "",
                "correct_index": -1,
                "challenge_type": "question_sans_reponse",
            }
        )
    return pd.DataFrame.from_records(challenges)


def cohen_kappa_from_review(
    reviews: pd.DataFrame, evaluator_a: str, evaluator_b: str, metric: str
) -> float:
    """Calcule le kappa de Cohen sur les notes completes de deux evaluateurs."""
    from sklearn.metrics import cohen_kappa_score

    columns = [evaluator_a, evaluator_b]
    if any(column not in reviews.columns for column in columns):
        raise KeyError(f"Colonnes d'evaluateurs absentes: {columns}")
    paired = reviews[columns].dropna()
    if paired.empty:
        raise ValueError(f"Aucune note commune renseignee pour {metric}.")
    return float(cohen_kappa_score(paired[evaluator_a], paired[evaluator_b]))
