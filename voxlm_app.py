import pandas as pd
import base64
import io
import json
import re
import logging
from typing import List, Dict, Any, Optional, Literal 
import os
from pydantic import BaseModel
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer, util #add to deployment update
import numpy as np #add to deployment update
from collections import Counter

#2.0 update add
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
import tempfile
import uuid
import subprocess
from pathlib import Path
import cv2
from faster_whisper import WhisperModel
import torch
import copy


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#Qwen3-VL-8B trannsformer engine script
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
logger.info("Model loading started...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
logger.info("Model loaded!")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

#System Prompt
SYSTEM_PROMPTS: Dict[str, str] = {
    "dentistry": (
        "You are an experienced dentistry examiner. "
        "You grade short-answer questions strictly according to the rubric, model answer, "
        "and the patterns in the human-marked examples.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include any explanations, commentary, or markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
        "- Do NOT use comments or any other non-JSON syntax.\n"),
    "medicine":(
        "You are an experienced examiner for medical students (medicine and surgery). "
        "You grade short-answer questions strictly according to the rubric, model answer, "
        "and the patterns in the human-marked examples.\n\n"
        "Pay close attention to legal reasoning, correct use of medical concepts, and application to patient scenarios. \n\n"
        "CRITICAL FORMAT INSTRUCTIONS: \n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include any explanations, commentary, or markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
        "- Do NOT use comments or any other non-JSON syntax.\n"    
    ),
    "law":(
        "You are an experienced law examiner. "
        "You grade short-answer questions strictly according to the rubric, model answer, "
        "and the patterns in the human-marked examples.\n\n"
        "Pay close attention to legal reasoning, correct use of legal concepts, and application to facts. \n\n"
        "CRITICAL FORMAT INSTRUCTIONS: \n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include any explanations, commentary, or markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
        "- Do NOT use comments or any other non-JSON syntax.\n"    
    ),
    "education":(
        "You are an experienced education examiner. "
        "You grade short-answer questions strictly according to the rubric, model answer, "
        "and the patterns in the human-marked examples.\n\n"
        "Pay close attention to pedagogical reasoning, correct use of educational theories, and application to teaching scenarios. \n\n"
        "CRITICAL FORMAT INSTRUCTIONS: \n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include any explanations, commentary, or markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
        "- Do NOT use comments or any other non-JSON syntax.\n"    
    )
}

#added to ensure consistency in marking
MARKING_ENFORCEMENT = (
    "\n\nCORE MARKING POLICY:\n"
    "- Use positive marking unless the rubric explicitly requires negative marking.\n"
    "- Award credit for every correct, creditworthy concept that appears in the student's answer.\n"
    "- Do NOT cancel a correct concept merely because it appears alongside extra irrelevant or incorrect wording.\n"
    "- If a student identifies the correct underlying mechanism but names an additional adjacent structure imprecisely, award the mechanism mark unless the rubric explicitly requires that exact structure only.\n"
    "- If a student writes a mixed phrase such as 'fractured or distorted framework', and 'distorted framework' matches the rubric, award the mark for distortion. The incorrect word 'fractured' should receive no credit, but it should not cancel the correct word unless the rubric explicitly says so.\n"
    "- If the rubric says an incorrect reason scores 0, this means that incorrect reason alone earns 0. It does not mean the whole answer or criterion scores 0 if a valid reason is also present.\n"
    "- Do not apply negative marking unless the rubric explicitly instructs you to do so.\n"
    "- Award marks item-by-item. The final score must equal the sum of the awarded item marks or subquestion marks.\n"
    "- Do not award half marks unless the rubric explicitly allows half marks, or the student gives a genuinely partial/incomplete version of a required concept.\n"
    "- If the rubric has discrete 1-mark points, each point should normally be scored as 0 or 1, not 0.5.\n"
)


#Prompt for Tab 2 (Summarization)
SUMMARY_SYSTEM_PROMPTS: Dict[str, str] = {
    "dentistry": (
        "You are an experienced dentistry teacher and assessment lead. "
        "You are given exact class statistics computed locally and representative evidence from students in each tier of performance "
        "from a graded SAQ cohort. Your job is to interpret those results using the question stem, "
        "model answer, rubric, and subquestion structure.\n\n"
        "You must identify what high, mid, and low performers appear to understand or misunderstand, "
        "and give practical next steps for teachers to better support their learning.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
    "medicine": (
    "You are an experienced medicine and surgery teacher and assessment lead. "
    "You are given exact class statistics computed locally and representative evidence from students in each tier of performance "
    "from a graded SAQ cohort. Your job is to interpret those results using the question stem, "
    "model answer, rubric, and subquestion structure.\n\n"
    "You must identify what high, mid, and low performers appear to understand or misunderstand, "
    "and give practical next steps for teachers to better support their learning.\n\n"
    "CRITICAL FORMAT INSTRUCTIONS:\n"
    "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
    "- Do NOT include markdown outside the JSON.\n"
    "- Use only double quotes for strings.\n"
    "- Do NOT use trailing commas.\n"
    ),
    "law": (
        "You are an experienced law teacher and assessment lead. "
        "You are given exact class statistics computed locally and representative evidence from students in each tier of performance "
        "from a graded SAQ cohort. Your job is to interpret those results using the question stem, "
        "model answer, rubric, and subquestion structure.\n\n"
        "You must identify what high, mid, and low performers appear to understand or misunderstand, "
        "and give practical next steps for teachers to better support their learning.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
    "education": (
        "You are an experienced education teacher and assessment lead. "
        "You are given exact class statistics computed locally and representative evidence from students in each tier of performance "
        "from a graded SAQ cohort. Your job is to interpret those results using the question stem, "
        "model answer, rubric, and subquestion structure.\n\n"
        "You must identify what high, mid, and low performers appear to understand or misunderstand, "
        "and give practical next steps for teachers to better support their learning.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Do NOT include markdown outside the JSON.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
}

def get_summary_system_prompt(discipline: str) -> str:
    key = (discipline or "dentistry").lower()
    return SUMMARY_SYSTEM_PROMPTS.get(key, SUMMARY_SYSTEM_PROMPTS["dentistry"])

def get_system_prompt(discipline: str) -> str:
    """Return the appropriate system prompt for the discipline."""
    key = (discipline or "dentistry").lower()
    return SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS["dentistry"]) + MARKING_ENFORCEMENT


#Prompts for Tab 3

SOLO_SYSTEM_PROMPTS: Dict[str, str] = {
    "dentistry": (
        "You are an assessment expert in dentistry and SOLO taxonomy. "
        "You analyse SAQ questions using the SOLO taxonomy and return only valid JSON.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
    "medicine": (
    "You are an assessment expert in medicine and surgery and SOLO taxonomy. "
    "You analyse SAQ questions using the SOLO taxonomy and return only valid JSON.\n\n"
    "CRITICAL FORMAT INSTRUCTIONS:\n"
    "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
    "- Use only double quotes for strings.\n"
    "- Do NOT use trailing commas.\n"
    ),
    "law": (
        "You are an assessment expert in law and SOLO taxonomy. "
        "You analyse SAQ questions using the SOLO taxonomy and return only valid JSON.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
    "education": (
        "You are an assessment expert in education and SOLO taxonomy. "
        "You analyse SAQ questions using the SOLO taxonomy and return only valid JSON.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
}

def get_solo_system_prompt(discipline: str) -> str:
    key = (discipline or "dentistry").lower()
    return SOLO_SYSTEM_PROMPTS.get(key, SOLO_SYSTEM_PROMPTS["dentistry"])

#Student report system prompt
STUDENT_REPORT_SYSTEM_PROMPTS: Dict[str, str] = {
    "dentistry": (
        "You write short, clear, student-friendly feedback for dentistry students. "
        "Write in plain English, avoid technical assessment jargon, and keep feedback concise. "
        "Return only valid JSON.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
    "medicine": (
    "You write short, clear, student-friendly feedback for medicine and surgery students. "
    "Write in plain English, avoid technical assessment jargon, and keep feedback concise. "
    "Return only valid JSON.\n\n"
    "CRITICAL FORMAT INSTRUCTIONS:\n"
    "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
    "- Use only double quotes for strings.\n"
    "- Do NOT use trailing commas.\n"
    ),
    "law": (
        "You write short, clear, student-friendly feedback for law students. "
        "Write in plain English, avoid technical assessment jargon, and keep feedback concise. "
        "Return only valid JSON.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
    "education": (
        "You write short, clear, student-friendly feedback for education students. "
        "Write in plain English, avoid technical assessment jargon, and keep feedback concise. "
        "Return only valid JSON.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- You MUST respond with a SINGLE valid JSON object and NOTHING ELSE.\n"
        "- Use only double quotes for strings.\n"
        "- Do NOT use trailing commas.\n"
    ),
}


def get_student_report_system_prompt(discipline: str) -> str:
    key = (discipline or "dentistry").lower()
    return STUDENT_REPORT_SYSTEM_PROMPTS.get(key, STUDENT_REPORT_SYSTEM_PROMPTS["dentistry"])


#Request and response schemas - pydantic
class Question(BaseModel):
    exam_id: Optional[str] = "EXAM"
    question_id: Optional[str] = "Q1"
    stem: str
    max_score: Optional[float] = None
    subquestions: Optional[List[Dict[str, Any]]] = []
    model_answer: Optional[str] = None
    rubric: Optional[str] = None 


class ResponseItem(BaseModel):
    response_id: str
    answers: Dict[str, str]


class FewShotItem(BaseModel):
    response_id: str
    answers: Dict[str, str]
    marker_score: float

class GradeRequest(BaseModel):
    question: Question
    few_shot: List[FewShotItem] = []
    student_response: ResponseItem
    has_subquestions: bool = True
    images: List[str] = []
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"
    challenge_mode: bool = False
    challenge_reason: Optional[str] = None
    original_total_score: Optional[float] = None
    original_sub_scores: Optional[Dict[str, Any]] = None


class GradeResult(BaseModel):
    response_id: str
    total_score: float
    sub_scores: Dict[str, Any]
    rationale: str
    feedback: Dict[str, str]
    confidence: float
    highlights: Dict[str, Dict[str, List[str]]]
    debug_prompt: Optional[str] = None
    challenge_review: Optional[str] = ""
    challenged: bool = False
    original_total_score: Optional[float] = None

    missing_key_point: str = ""
    needs_review: bool = False
    review_reasons: List[str] = []

    marking_breakdown: Optional[List[Dict[str, Any]]] = []

#pydantic schema for batch summary
class BatchSummaryRequest(BaseModel):
    csv_text: str
    question: Question
    has_subquestions: bool = True
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"
    max_examples_per_tier: int = 5


class BatchSummaryResult(BaseModel):
    total_students: int
    scored_students: int
    class_average: float
    median_score: float
    max_score: float
    min_score: float
    std_score: float
    tier_thresholds: Dict[str, str]
    tier_counts: Dict[str, int]
    score_distribution: Dict[str, int]
    overall_subquestion_stats: Dict[str, Any]
    tier_summaries: Dict[str, str]
    strengths: List[str]
    common_misconceptions: List[Any]
    out_of_scope_points: List[Any] = []
    weak_areas: List[str]
    teacher_next_steps: List[str]
    narrative_summary: str
    subquestion_diagnostics: Dict[str, Any]
    debug_prompt: Optional[str] = None

#pydantic schema for batch norm reference grading

class BatchNormReferenceRequest(BaseModel):
    csv_text: str
    question: Question
    has_subquestions: bool = True
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"

class BatchNormReferenceResult(BaseModel):
    teacher_rows: List[Dict[str, Any]]
    diagnostic_rows: List[Dict[str, Any]]

#pydantic schema for student statistics 
class StudentReportRequest(BaseModel):
    criterion_csv_text: str
    norm_csv_text: str
    question: Question
    has_subquestions: bool = True
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"
    max_total_bullets_per_student: int = 3


class StudentReportItem(BaseModel):
    student_id: str
    display_name: str
    overall_summary: str
    strong_areas: List[str]
    weak_areas: List[str]
    report_text: Optional[str] = None


class StudentReportBatchResult(BaseModel):
    solo_question_analysis: Dict[str, Any]
    reports: List[StudentReportItem]
    debug_prompt_sample: Optional[str] = None

#pydantic schema for handwriting transcription 
class HandwritingTranscriptionRequest(BaseModel):
    image: str
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"


class HandwritingTranscriptionResult(BaseModel):
    transcription: str
    confidence: float
    debug_prompt: Optional[str] = None

#pydantic schema for chat interface with students
class VoxChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class VoxChatRequest(BaseModel):
    question: Question
    student_response: ResponseItem
    grade_result: Dict[str, Any]
    chat_history: List[VoxChatMessage] = []
    user_message: str
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"


class VoxChatResult(BaseModel):
    assistant_message: str
    debug_prompt: Optional[str] = None

#pydantic schema for refining model answer
class RefineModelAnswerRequest(BaseModel):
    question: Question
    discipline: Literal["dentistry", "medicine", "law", "education"] = "dentistry"


class RefineModelAnswerResult(BaseModel):
    rating_score: float
    rating_label: str
    strengths: List[str]
    issues: List[str]
    rewritten_model_answer: str
    suggested_marking_structure: str
    debug_prompt: Optional[str] = None


#Version 2.0 Schema
class VideoMCQGenerateResult(BaseModel):
    video_id: str
    filename: str
    duration_seconds: float
    transcript_summary: str
    transcript_segments: List[Dict[str, Any]]
    frame_summaries: List[Dict[str, Any]]
    teaching_segments: List[Dict[str, Any]]
    pre_question: Dict[str, Any]
    embedded_questions: List[Dict[str, Any]]
    warnings: List[str] = []
    debug_prompt: Optional[str] = None


#text analysis helpers
WORD_RE = re.compile(r"\b[\w'-]+\b")

def tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return WORD_RE.findall(str(text).lower())

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[.!?]+", str(text))
    return [p.strip() for p in parts if p.strip()]

def moving_ttr(tokens: List[str], window: int = 25) -> float:
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return len(set(tokens)) / max(1, len(tokens))
    vals = []
    for i in range(len(tokens) - window + 1):
        chunk = tokens[i:i+window]
        vals.append(len(set(chunk)) / window)
    return float(sum(vals) / len(vals)) if vals else 0.0

def repetition_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))

def percentile_rank_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return s.rank(method="average", pct=True) * 100.0

def extract_highlight_pattern_counts(
    scored_df: pd.DataFrame,
    category: str,
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    """
    Count how many students have each highlighted segment.

    category can be:
    - "misconception"
    - "out_of_scope"

    Backward compatibility:
    - old "incorrect" highlights are treated as "misconception".
    """
    if "highlights_json" not in scored_df.columns:
        return []

    counter: Counter = Counter()
    display_lookup: Dict[str, str] = {}

    total_students = int(len(scored_df))
    if total_students == 0:
        return []

    for _, row in scored_df.iterrows():
        raw = row.get("highlights_json", "")
        if pd.isna(raw) or not str(raw).strip():
            continue

        try:
            h_obj = json.loads(str(raw))
        except Exception:
            continue

        if not isinstance(h_obj, dict):
            continue

        student_segments = set()

        for _, sub_h in h_obj.items():
            if not isinstance(sub_h, dict):
                continue

            if category == "misconception":
                segments = (
                    sub_h.get("misconception", [])
                    or sub_h.get("incorrect", [])
                    or []
                )
            else:
                segments = sub_h.get(category, []) or []

            for seg in segments:
                seg_text = str(seg or "").strip()
                if len(seg_text) < 3:
                    continue

                key = re.sub(r"\s+", " ", seg_text.lower()).strip()
                if not key:
                    continue

                student_segments.add(key)
                display_lookup.setdefault(key, seg_text)

        for key in student_segments:
            counter[key] += 1

    rows = []
    for key, count in counter.most_common(max_items):
        rows.append({
            "point": display_lookup.get(key, key),
            "student_count": int(count),
            "percent_students": round((count / total_students) * 100.0, 1),
        })

    return rows


#sentence embedding and similarity helpers for semantic comparison to reference answers
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def semantic_similarity_to_reference(texts: List[str], reference: str) -> List[Optional[float]]:
    if not reference or not str(reference).strip():
        return [None] * len(texts)

    all_texts = [reference] + [t or "" for t in texts]
    emb = embed_model.encode(all_texts, convert_to_tensor=True)
    ref_emb = emb[0]
    text_embs = emb[1:]
    sims = util.cos_sim(text_embs, ref_emb).cpu().numpy().reshape(-1)

    return [round(float((x + 1.0) / 2.0 * 100.0), 2) for x in sims]

#uild norm reference text for grading
def build_norm_reference_text(question: Question) -> str:
    parts = []

    if question.stem:
        parts.append(f"Question stem:\n{question.stem.strip()}")

    if question.model_answer:
        parts.append(f"Model answer:\n{question.model_answer.strip()}")

    if question.rubric:
        parts.append(f"Rubric:\n{question.rubric.strip()}")

    for sq in question.subquestions or []:
        sid = str(sq.get("id", "")).strip()
        prompt = str(sq.get("prompt", "")).strip()
        rubric = str(sq.get("rubric", "") or "").strip()
        block = []
        if sid:
            block.append(f"Subquestion {sid}")
        if prompt:
            block.append(f"Prompt: {prompt}")
        if rubric:
            block.append(f"Rubric: {rubric}")
        if block:
            parts.append("\n".join(block))

    return "\n\n".join(p for p in parts if p.strip())

#qualitative assignment helpers
def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def percentile_band_label(p: Any) -> str:
    p = safe_float(p)
    if p is None:
        return "Unavailable"
    if p >= 90:
        return "Among the strongest in the class"
    if p >= 75:
        return "Stronger than most of the class"
    if p >= 60:
        return "Above the class midpoint"
    if p >= 40:
        return "Around the class midpoint"
    if p >= 25:
        return "Below the class midpoint"
    if p >= 10:
        return "Weaker than most of the class"
    return "Among the weakest in the class"


def length_comparison_label(p: Any, score_eff_p: Any = None) -> str:
    p = safe_float(p)
    eff = safe_float(score_eff_p)
    if p is None:
        return "Response length unavailable"

    if p >= 75:
        if eff is not None and eff < 40:
            return "Longer than most responses, without a comparable gain in marks"
        return "Longer than most responses"
    if p >= 40:
        return "About typical in length for the class"
    if eff is not None and eff >= 60:
        return "Shorter than most responses, but relatively efficient"
    return "Shorter than most responses"


def sentence_style_label(p: Any) -> str:
    p = safe_float(p)
    if p is None:
        return "Sentence style unavailable"
    if p >= 75:
        return "Uses more developed sentence structure than most of the class"
    if p >= 40:
        return "Sentence structure is broadly typical for the class"
    return "Sentence structure is simpler than most of the class"


def vocabulary_label(p: Any) -> str:
    p = safe_float(p)
    if p is None:
        return "Vocabulary comparison unavailable"
    if p >= 75:
        return "Uses more varied vocabulary than most of the class"
    if p >= 40:
        return "Vocabulary variety is typical for the class"
    return "Vocabulary is less varied than most of the class"


def conceptual_understanding_label(p: Any) -> str:
    p = safe_float(p)
    if p is None:
        return "Conceptual comparison unavailable"
    if p >= 75:
        return "Shows stronger conceptual understanding than most of the class"
    if p >= 40:
        return "Shows a level of conceptual understanding similar to much of the class"
    return "Shows weaker conceptual understanding than most of the class"


def precision_expression_label(p: Any) -> str:
    p = safe_float(p)
    if p is None:
        return "Precision comparison unavailable"
    if p >= 75:
        return "Expresses ideas more precisely than most of the class"
    if p >= 40:
        return "Precision of expression is broadly typical for the class"
    return "Ideas are less precisely expressed than in stronger class responses"


def conciseness_label(p: Any) -> str:
    p = safe_float(p)
    if p is None:
        return "Conciseness comparison unavailable"
    if p >= 75:
        return "More concise than most responses while still gaining marks"
    if p >= 40:
        return "Conciseness is about typical for the class"
    return "Less concise than most responses"


def build_strengths_relative_to_class(row: pd.Series) -> str:
    strengths = []

    if safe_float(row.get("conceptual_accuracy_index")) is not None and float(row["conceptual_accuracy_index"]) >= 75:
        strengths.append("conceptual understanding is stronger than most of the class")

    if safe_float(row.get("lexical_richness_percentile")) is not None and float(row["lexical_richness_percentile"]) >= 75:
        strengths.append("vocabulary is more varied than in most responses")

    if safe_float(row.get("conciseness_index")) is not None and float(row["conciseness_index"]) >= 75:
        strengths.append("the response is relatively concise for the amount of credit gained")

    if safe_float(row.get("avg_sentence_length_percentile")) is not None and float(row["avg_sentence_length_percentile"]) >= 75:
        strengths.append("sentence structure is more developed than in many class responses")

    return "; ".join(strengths) if strengths else "No especially strong relative features identified."


def build_improvement_relative_to_class(row: pd.Series) -> str:
    issues = []

    if safe_float(row.get("conceptual_accuracy_index")) is not None and float(row["conceptual_accuracy_index"]) < 40:
        issues.append("conceptual understanding appears weaker than much of the class")

    if safe_float(row.get("conceptual_precision_index")) is not None and float(row["conceptual_precision_index"]) < 40:
        issues.append("ideas could be expressed more precisely")

    if safe_float(row.get("conciseness_index")) is not None and float(row["conciseness_index"]) < 40:
        issues.append("the response could be more concise")

    if safe_float(row.get("lexical_richness_percentile")) is not None and float(row["lexical_richness_percentile"]) < 40:
        issues.append("vocabulary is less varied than in many responses")

    return "; ".join(issues) if issues else "No major relative weaknesses identified."


def build_teacher_summary(row: pd.Series) -> str:
    return (
        f"This response is {str(row.get('overall_class_position', '')).lower()}. "
        f"It is {str(row.get('response_length_comparison', '')).lower()}. "
        f"It {str(row.get('conceptual_understanding_comparison', '')).lower()}. "
        f"Vocabulary use is described as: {str(row.get('vocabulary_variety_comparison', '')).lower()}. "
        f"Overall, {str(row.get('conciseness_comparison', '')).lower()}."
    )

#norm reference grading helpers
def build_norm_reference_dfs(csv_text: str, question: Question) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(io.StringIO(csv_text))

    response_cols = [c for c in df.columns if str(c).startswith("STUDENT RESPONSES")]
    if not response_cols:
        raise ValueError("CSV must contain a column whose name starts with 'STUDENT RESPONSES'.")
    if "total_score" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'total_score' column.")

    response_col = response_cols[0]
    work = df.copy()

    work["_response_text"] = work[response_col].fillna("").astype(str)
    work["_tokens"] = work["_response_text"].apply(tokenize_words)
    work["_sentences"] = work["_response_text"].apply(split_sentences)

    work["word_count"] = work["_tokens"].apply(len)
    work["sentence_count"] = work["_sentences"].apply(len)
    work["avg_sentence_length"] = work.apply(
        lambda r: round(r["word_count"] / r["sentence_count"], 2) if r["sentence_count"] else 0.0,
        axis=1,
    )
    work["lexical_richness"] = work["_tokens"].apply(lambda t: round(moving_ttr(t), 4))
    work["long_word_ratio"] = work["_tokens"].apply(
        lambda t: round(sum(1 for x in t if len(x) >= 7) / len(t), 4) if t else 0.0
    )
    work["repetition_ratio"] = work["_tokens"].apply(lambda t: round(repetition_ratio(t), 4))

    scores = pd.to_numeric(work["total_score"], errors="coerce")
    qmax = question.max_score if question.max_score is not None else None

    work["grade_percentile"] = percentile_rank_series(scores).round(1)

    if qmax is not None and float(qmax) > 0:
        work["score_percent"] = ((scores / float(qmax)) * 100.0).round(2)
    else:
        work["score_percent"] = work["grade_percentile"]

    reference_text = build_norm_reference_text(question)
    sims = semantic_similarity_to_reference(work["_response_text"].tolist(), reference_text)
    work["semantic_similarity_to_model_answer"] = sims

    for col in [
        "word_count",
        "avg_sentence_length",
        "lexical_richness",
        "long_word_ratio",
        "repetition_ratio",
    ]:
        work[f"{col}_percentile"] = percentile_rank_series(work[col]).round(1)

    work["score_per_100_words"] = work.apply(
        lambda r: round((float(r["total_score"]) / r["word_count"]) * 100.0, 2)
        if pd.notna(r["total_score"]) and r["word_count"] > 0 else np.nan,
        axis=1,
    )
    work["score_per_100_words_percentile"] = percentile_rank_series(work["score_per_100_words"]).round(1)

    work["semantic_similarity_percentile"] = percentile_rank_series(
        pd.Series(work["semantic_similarity_to_model_answer"])
    ).round(1)

    work["language_complexity_index"] = (
        0.45 * work["avg_sentence_length_percentile"].fillna(0) +
        0.35 * work["lexical_richness_percentile"].fillna(0) +
        0.20 * work["long_word_ratio_percentile"].fillna(0)
    ).round(1)

    sim_component = work["semantic_similarity_percentile"].fillna(work["score_percent"])

    work["conceptual_accuracy_index"] = (
        0.70 * work["score_percent"].fillna(0) +
        0.30 * sim_component.fillna(0)
    ).round(1)

    work["conceptual_precision_index"] = (
        0.50 * work["score_per_100_words_percentile"].fillna(0) +
        0.30 * sim_component.fillna(0) +
        0.20 * (100.0 - work["repetition_ratio_percentile"].fillna(0))
    ).round(1)

    work["conciseness_index"] = (
        0.70 * work["score_per_100_words_percentile"].fillna(0) +
        0.30 * (100.0 - work["word_count_percentile"].fillna(0))
    ).round(1)

    work["overall_class_position"] = work["grade_percentile"].apply(percentile_band_label)
    work["response_length_comparison"] = work.apply(
        lambda r: length_comparison_label(r.get("word_count_percentile"), r.get("score_per_100_words_percentile")),
        axis=1,
    )
    work["sentence_style_comparison"] = work["language_complexity_index"].apply(sentence_style_label)
    work["vocabulary_variety_comparison"] = work["lexical_richness_percentile"].apply(vocabulary_label)
    work["conceptual_understanding_comparison"] = work["conceptual_accuracy_index"].apply(conceptual_understanding_label)
    work["precision_of_expression_comparison"] = work["conceptual_precision_index"].apply(precision_expression_label)
    work["conciseness_comparison"] = work["conciseness_index"].apply(conciseness_label)
    work["strengths_relative_to_class"] = work.apply(build_strengths_relative_to_class, axis=1)
    work["areas_for_improvement_relative_to_class"] = work.apply(build_improvement_relative_to_class, axis=1)
    work["norm_referenced_summary"] = work.apply(build_teacher_summary, axis=1)

    work["result_type"] = "norm_referenced"

    diagnostic_cols = []
    if "STUDENT_ID" in work.columns:
        diagnostic_cols.append("STUDENT_ID")

    diagnostic_cols += [
        response_col,
        "total_score",
        "score_percent",
        "grade_percentile",
        "word_count",
        "word_count_percentile",
        "sentence_count",
        "avg_sentence_length",
        "avg_sentence_length_percentile",
        "lexical_richness",
        "lexical_richness_percentile",
        "long_word_ratio",
        "long_word_ratio_percentile",
        "repetition_ratio",
        "repetition_ratio_percentile",
        "score_per_100_words",
        "score_per_100_words_percentile",
        "semantic_similarity_to_model_answer",
        "semantic_similarity_percentile",
        "language_complexity_index",
        "conceptual_accuracy_index",
        "conceptual_precision_index",
        "conciseness_index",
        "result_type",
    ]

    teacher_cols = []
    if "STUDENT_ID" in work.columns:
        teacher_cols.append("STUDENT_ID")

    teacher_cols += [
        response_col,
        "total_score",
        "overall_class_position",
        "response_length_comparison",
        "sentence_style_comparison",
        "vocabulary_variety_comparison",
        "conceptual_understanding_comparison",
        "precision_of_expression_comparison",
        "conciseness_comparison",
        "strengths_relative_to_class",
        "areas_for_improvement_relative_to_class",
        "norm_referenced_summary",
        "result_type",
    ]

    diagnostic_df = work[diagnostic_cols].copy()
    teacher_df = work[teacher_cols].copy()

    return teacher_df, diagnostic_df

# helper to simplify student report text
def simplify_student_report_text(text: str) -> str:
    replacements = {
        "criterion-referenced": "",
        "norm-referenced": "compared with the class",
        "SOLO taxonomy": "",
        "prestructural": "showing very limited understanding",
        "unistructural": "identifying one key point",
        "multistructural": "including several relevant points",
        "relational": "linking ideas together clearly",
        "extended abstract": "applying ideas more broadly",
        "conceptual precision": "clear and accurate wording",
        "conceptual understanding": "understanding of the topic",
    }

    out = str(text or "")
    for src, dst in replacements.items():
        out = out.replace(src, dst)

    return " ".join(out.split())

#helper functions for review
def flatten_response_text(resp: ResponseItem) -> str:
    if not resp.answers:
        return ""
    return "\n".join(str(v or "") for v in resp.answers.values())


def compute_needs_review(req: GradeRequest, pred: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons: List[str] = []

    answer_text = flatten_response_text(req.student_response)
    tokens = tokenize_words(answer_text)

    total_score = safe_float(pred.get("total_score")) or 0.0
    confidence = safe_float(pred.get("confidence")) or 0.0
    qmax = safe_float(req.question.max_score)

    if confidence < 60:
        reasons.append("Low grading confidence")

    if len(tokens) == 0:
        reasons.append("Blank or near-blank response")

    if qmax is not None and qmax > 0:
        if len(tokens) <= 12 and total_score >= 0.7 * qmax:
            reasons.append("Very short response received a relatively high mark")

        if len(tokens) >= 120 and total_score <= 0.25 * qmax:
            reasons.append("Long response received a relatively low mark")

    if req.challenge_mode:
        reasons.append("Grade was reviewed through challenge flow")

    highlights = pred.get("highlights", {}) or {}
    has_positive_evidence = False
    if isinstance(highlights, dict):
        for _, h in highlights.items():
            if isinstance(h, dict) and h.get("correct"):
                has_positive_evidence = True
                break

    if total_score > 0 and not has_positive_evidence:
        reasons.append("No positive evidence highlights were returned for a non-zero mark")

    return bool(reasons), reasons

#format score for display, avoiding unnecessary decimals
def fmt_score(x: Any) -> str:
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(round(f, 2)).rstrip("0").rstrip(".")
    except Exception:
        return str(x)

#helper to sync rationale with final score and avoid contradictions or stale totals in rationale text
def sync_rationale_with_final_score(req: GradeRequest, pred: Dict[str, Any]) -> None:
    """
    Ensure rationale does not contain a stale or contradictory Total: x/y statement.

    If the model wrote a different total in the rationale than the final backend score,
    rebuild a short rationale from marking_breakdown where possible.
    """
    try:
        final_total = float(pred.get("total_score", 0.0))
    except Exception:
        final_total = 0.0

    qmax = req.question.max_score
    qmax_text = fmt_score(qmax) if qmax is not None else ""

    rationale = str(pred.get("rationale", "") or "").strip()

    # Detect stale "Total: x/y" inside model rationale
    stale_total_found = False
    stale_total_mismatch = False

    m = re.search(
        r"\bTotal\s*:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
        rationale,
        flags=re.I,
    )

    if m:
        stale_total_found = True
        try:
            stated_total = float(m.group(1))
            if abs(stated_total - final_total) > 1e-6:
                stale_total_mismatch = True
        except Exception:
            stale_total_mismatch = True

    breakdown = pred.get("marking_breakdown", []) or []

    # If rationale total contradicts backend total, rebuild from marking_breakdown
    if stale_total_mismatch and isinstance(breakdown, list) and breakdown:
        credited = []
        not_credited = []

        for item in breakdown:
            if not isinstance(item, dict):
                continue

            try:
                awarded = float(item.get("awarded", 0.0) or 0.0)
            except Exception:
                awarded = 0.0

            criterion = str(item.get("criterion", "") or "").strip()
            evidence = str(item.get("student_evidence", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()

            if awarded > 0:
                if evidence:
                    credited.append(f'"{evidence}"')
                elif criterion:
                    credited.append(criterion)
            else:
                if criterion:
                    not_credited.append(criterion)
                elif reason:
                    not_credited.append(reason)

        new_rationale_parts = []

        if credited:
            new_rationale_parts.append(
                "Credit awarded for: " + "; ".join(credited[:5]) + "."
            )
        else:
            new_rationale_parts.append(
                "No clearly creditworthy points were identified in the response."
            )

        if not_credited and final_total < float(qmax or final_total or 0):
            new_rationale_parts.append(
                "Main missing or unsupported point: " + not_credited[0] + "."
            )

        if qmax is not None:
            new_rationale_parts.append(
                f"Final score: {fmt_score(final_total)}/{qmax_text}."
            )
        else:
            new_rationale_parts.append(
                f"Final score: {fmt_score(final_total)}."
            )

        pred["rationale"] = " ".join(new_rationale_parts).strip()
        return

    # avoid duplicate total score lines in rationale
    rationale = re.sub(
        r"\bTotal\s*:\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\.?",
        "",
        rationale,
        flags=re.I,
    ).strip()

    # Avoid duplicate final score lines
    rationale = re.sub(
        r"\bFinal score\s*:\s*\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\.?",
        "",
        rationale,
        flags=re.I,
    ).strip()

    if qmax is not None:
        final_line = f"Final score: {fmt_score(final_total)}/{qmax_text}."
    else:
        final_line = f"Final score: {fmt_score(final_total)}."

    if rationale:
        pred["rationale"] = f"{rationale} {final_line}".strip()
    else:
        pred["rationale"] = final_line

#Prompting and model call
def build_rubric_summary(question: Question) -> str:
    """
    Text summary of question, model answer, rubric, and subquestions.
    """
    lines = []
    lines.append("Question metadata:")
    lines.append(f"- exam_id: {question.exam_id}")
    lines.append(f"- question_id: {question.question_id}")
    if question.max_score is not None:
        lines.append(f"- total_max_score: {question.max_score}")
    lines.append("")

    lines.append("Question stem:")
    lines.append(question.stem)
    lines.append("")

    if question.model_answer:
        lines.append("Model answer (ideal solution):")
        lines.append(question.model_answer.strip())
        lines.append("")

    if question.rubric:
        lines.append("Global marking rubric / scheme:")
        lines.append(question.rubric.strip())
        lines.append("")

    subquestions = question.subquestions or []
    if subquestions:
        lines.append("Subquestions and marking rubric:")
        for sq in subquestions:
            sid = str(sq.get("id", ""))
            prompt = sq.get("prompt", "")
            max_s = sq.get("max_score", None)
            rubric = (sq.get("rubric") or "").strip()

            lines.append("")
            if max_s is not None:
                lines.append(f"({sid}) max {max_s}")
            else:
                lines.append(f"({sid})")
            lines.append(f"Prompt: {prompt}")
            if rubric:
                lines.append(f"Rubric: {rubric}")

    lines.append("")
    lines.append(
        "The total score for the whole question ranges from 0 up to the total_max_score "
        "(if provided). Subquestion scores must be between 0 and that subquestion's max_score. "
        "Use the marking units in the rubric. If the rubric lists discrete 1-mark points, "
        "score each point as 0 or 1 unless the rubric explicitly allows partial credit. "
        "Half marks should be used only when the rubric allows them or when a concept is genuinely partially expressed."
    )


    return "\n".join(lines)


def format_few_shot_example(resp: FewShotItem, sub_ids: List[str]) -> str:
    """
    Format one human-marked example into text for the prompt.
    """
    lines = [
        "Example",
        f"response_id: {resp.response_id}",
        "Student answers:",
    ]
    ans = resp.answers
    if sub_ids:
        for sid in sub_ids:
            lines.append(f"({sid}) {ans.get(sid, '').strip()}")
    else:
        lines.append(f"(overall) {ans.get('overall', '').strip()}")
    lines.append(f"Human total score: {resp.marker_score}")
    return "\n".join(lines)


def build_user_text(
    question: Question,
    few_shot_list: List[FewShotItem],
    target_resp: ResponseItem,
    has_subquestions: bool,
    challenge_mode: bool = False,
    challenge_reason: Optional[str] = None,
    original_total_score: Optional[float] = None,
    original_sub_scores: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build full user prompt text for Qwen, including instructions and JSON schema.
    """
    rubric_summary = build_rubric_summary(question)
    subquestions = question.subquestions or []
    sub_ids = [str(sq.get("id")) for sq in subquestions]

    parts = [rubric_summary, "\n\nBelow are EXAMPLES of student responses and their HUMAN-MARKED total scores:\n"]

    for ex in few_shot_list:
        parts.append(format_few_shot_example(ex, sub_ids))
        parts.append("")

    ans = target_resp.answers
    total_max = question.max_score

    target_lines = [
        "Now, here is a NEW student response that you must grade:\n",
        "Target response",
        f"response_id: {target_resp.response_id}",
        "Student answers:",
    ]

    if has_subquestions and sub_ids:
        for sid in sub_ids:
            target_lines.append(f"({sid}) {ans.get(sid, '').strip()}")
    else:
        target_lines.append(f"(overall) {ans.get('overall', '').strip()}")

    target_lines.append("")
    target_lines.append("Task:")
    target_lines.append(
        "Using ONLY the question, model answer, rubric, and the patterns in the examples, "
        "predict a fair score for this target response."
    )
    if total_max is not None:
        target_lines.append(
            f"The total_score must be between 0.0 and {total_max}. "
            "Use the mark increments implied by the rubric. If the rubric has discrete 1-mark items, "
            "award 0 or 1 for each item. Use half-point steps only when the rubric allows partial credit "
            "or the answer is genuinely partially correct."
        )

    else:
        target_lines.append(
            "The total_score must be a non-negative number, and you may use half-point steps when partially correct."
        )

    target_lines.append("")
    if has_subquestions and sub_ids:
        target_lines.append("Subquestion scoring rules:")
        target_lines.append(
            "For each subquestion, you must assign a sub-score between 0 and its max_score. "
            "Use the marking increments implied by the rubric. If the subquestion has discrete 1-mark points, score each as 0 or 1. Use half marks only when justified by the rubric or genuine partial expression."
        )
        sq_score_lines = []
        for sq in subquestions:
            sid = str(sq.get("id"))
            max_s = sq.get("max_score", None)
            if max_s is not None:
                sq_score_lines.append(f"- {sid}: from 0 up to {max_s}, using half-point steps if justified.")
            else:
                sq_score_lines.append(f"- {sid}: score according to rubric (max_score not specified).")
        target_lines.append("\n".join(sq_score_lines))
        target_lines.append("Ensure that total_score MUST equal the sum of all sub_scores. This is a strict requirement!")
    else:
        target_lines.append(
            "This question is treated as a single overall answer with no subquestions. "
            "In that case, set sub_scores to an empty object {} and focus on total_score and overall feedback."
        )

    target_lines.append("")
    target_lines.append("IMPORTANT RULES:")
    target_lines.append(
        "- You MUST grade only what this student actually wrote. "
        "Do not assume or invent missing content."
    )
    target_lines.append(
        "- Use POSITIVE MARKING. Award marks for correct creditworthy ideas that are present."
    )
    target_lines.append(
        "- Do NOT withhold a mark for a correct idea simply because the student also included extra irrelevant or incorrect wording."
    )
    target_lines.append(
        "- A mixed phrase can contain both a correct and an incorrect element. Award credit for the correct element and ignore the incorrect element unless the rubric explicitly requires negative marking."
    )
    target_lines.append(
        "- If the rubric states that an incorrect reason scores 0, this means that the incorrect reason itself earns no credit. It does NOT cancel other correct reasons in the same response."
    )
    target_lines.append(
        "- Example rule: if the rubric accepts 'distortion/deformation of framework' and the student writes 'fractured or distorted framework', award credit for 'distorted framework'. The word 'fractured' earns no credit but does not cancel 'distorted'."
    )
    target_lines.append(
        "- Do not give unexplained half marks. If rubric points are listed as 1 mark each, award 1 for a clearly present correct point and 0 if absent, unless the rubric explicitly allows partial credit."
    )
    target_lines.append(
        "- The final numerical score must exactly match the marks justified in the rationale. If the rationale identifies two 1-mark points, the score must be at least 2."
    )

    target_lines.append(
        "- If an answer is blank, contains only spaces, or is obviously non‑informative "
        "(e.g. '-', 'N/A', 'nil'), you MUST give 0 marks for that subquestion (if any)."
    )
    target_lines.append(
        "- In your rationale you MUST quote the exact phrases from the student's answer that justify "
        "giving marks, or explicitly say 'no answer given' if blank."
        "Rationale scores must match total score and sum of all sub_scores. This is a strict requirement!"
    )

    target_lines.append("")
    target_lines.append(
        "In addition to scores, you MUST provide:\n"
        "- A brief final explanation (rationale) of how you arrived at the score.\n"
        "- The rationale MUST be short: maximum 80 words.\n"
        "-Come up with a straightforward rationale. Do not show your full thought process or all the details of the rubric. Just give a clear and concise explanation of the main reasons for the score you assigned, quoting specific phrases from the student's answer that justify the score.\n"
        "- Brief feedback for each subquestion (or overall) on where to improve.\n"
        "- A single short field called missing_key_point that states the most important creditworthy idea the student failed to include. If nothing major is missing, return an empty string.\n"
        "- A confidence value between 0 and 100 (e.g. 80) indicating how confident you are in your grading.\n"
        "- Highlighted response segments:\n"
        "  * For each subquestion, list exact short segments copied from the student's answer.\n"
        "  * correct: segments that are accurate and creditworthy.\n"
        "  * out_of_scope: segments that may be true or understandable but do not answer the question or do not earn credit under the rubric.\n"
        "  * misconception: segments that show an incorrect concept, wrong reasoning, or factual error.\n"
        "  * If a phrase contains both correct and incorrect parts, separate them where possible. For example, in 'fractured or distorted framework', list 'distorted framework' as correct and 'fractured' as misconception if fracture is not accepted.\n"
        "  * Do not let a misconception highlight cancel a correct highlight unless the rubric explicitly requires negative marking.\n"
        "  * uncertain: segments that are ambiguous, partially correct, unclear, or difficult to interpret.\n"
        "  * Do not invent segments. Each listed segment should appear in the student's original answer."
        
    )

    if challenge_mode:
        target_lines.append("")
        target_lines.append("CHALLENGE REVIEW MODE:")
        if original_total_score is not None:
            target_lines.append(f"- Original awarded total_score: {original_total_score}")
        if original_sub_scores:
            target_lines.append(f"- Original awarded sub_scores: {json.dumps(original_sub_scores)}")
        target_lines.append(f"- Student's challenge reason: {(challenge_reason or '').strip()}")

        target_lines.append(
            "- Re-mark the ORIGINAL submitted answer only. "
            "The challenge reason may justify extra credit only if it points to something already present in the original answer, "
            "or to a valid rubric-based interpretation of what the student actually wrote."
        )
        target_lines.append(
            "- Do NOT award extra marks for new facts, new explanations, or new claims that appear only in the challenge reason. "
            "Remember that students can be very tricky and may try to game the challenge system by writing a perfect answer in the challenge reason. "
            "All grading must be based on the original submitted answer, and the challenge reason can only be used to justify awarding marks that were already deserved based on the original answer."
        )
        target_lines.append(
            "- Do NOT reduce the student's original score because of the challenge. "
            "The revised score must stay the same or increase."
        )
        target_lines.append(
            "- In challenge_review, explain clearly whether the challenge reason was accepted or rejected, and why."
        )

    target_lines.append("")
    target_lines.append(
        "Before assigning the final score, create marking_breakdown entries for the main rubric points. "
        "Each entry must show the criterion, exact student evidence if present, awarded mark, maximum mark, and short reason. "
        "The total_score must equal the sum of awarded marks in marking_breakdown, unless subquestions are used, in which case total_score must equal the sum of sub_scores."
    )
    target_lines.append(
        "Return your answer as a single valid JSON object with this exact structure:\n\n"
        "{\n"
        '  \"response_id\": string,\n'
        '  \"total_score\": number,\n'
        '  \"sub_scores\": {\n'
        '    \"<sub_id>\": number,\n'
        "    ...\n"
        "  },\n"
        '  \"rationale\": string,\n'
        '  \"marking_breakdown\": [\n'
        '    {\n'
        '      \"criterion\": string,\n'
        '      \"student_evidence\": string,\n'
        '      \"awarded\": number,\n'
        '      \"max\": number,\n'
        '      \"reason\": string\n'
        '    }\n'
        '  ],\n'
        '  \"feedback\": {\n'
        '    \"<sub_id>\": string,\n'
        '    \"_overall\"?: string\n'
        "  },\n"
        '  \"missing_key_point\": string,\n'
        '  \"confidence\": number,\n'
        '  \"challenge_review\": string,\n'
        '  \"highlights\": {\n'
        '    \"<sub_id>\": {\n'
        '      \"correct\": [string, ...],\n'
        '      \"out_of_scope\": [string, ...],\n'
        '      \"misconception\": [string, ...],\n'
        '      \"uncertain\": [string, ...]\n'
        "    },\n"
        "    ...\n"
        "  }\n"
        "}\n\n"
        "Do not include any text before or after the JSON."
    )

    if challenge_mode:
        target_lines.append("Set challenge_review to a short explanation of whether the challenge was accepted or rejected.")
    else:
        target_lines.append('Set challenge_review to an empty string "".')

    parts.append("\n".join(target_lines))
    return "\n".join(parts)


def build_messages(
    question: Question,
    exam_images: List[Image.Image],
    few_shot_list: List[FewShotItem],
    target_resp: ResponseItem,
    has_subquestions: bool,
    discipline: str,
    challenge_mode: bool = False,
    challenge_reason: Optional[str] = None,
    original_total_score: Optional[float] = None,
    original_sub_scores: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    user_text = build_user_text(
    question,
    few_shot_list,
    target_resp,
    has_subquestions,
    challenge_mode=challenge_mode,
    challenge_reason=challenge_reason,
    original_total_score=original_total_score,
    original_sub_scores=original_sub_scores,
)


    user_content = []
    for img in exam_images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_text})

    system_prompt=get_system_prompt(discipline)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


def call_qwen_vl(messages, max_new_tokens: int = 1000) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    text = outputs[0]

    try:
        del inputs
        del generated_ids
        del generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return text


#helper to build messages for Vox-LM SAQ feedback assistant
def build_voxlm_chat_messages(req: VoxChatRequest) -> tuple[List[Dict[str, Any]], str]:
    system_prompt = (
        f"You are Vox-LM, an educational feedback assistant for {req.discipline} students. "
        "You must always identify yourself as Vox-LM if asked who you are. "
        "Your role is to help the student understand the feedback, rubric expectations, and how to improve.\n\n"
        "Important rules:\n"
        "- You must not change the grade.\n"
        "- You must not promise that the grade will change.\n"
        "- You may explain the existing feedback in clearer language.\n"
        "- You may help the student understand what was missing.\n"
        "- You may suggest how the student could improve in future answers.\n"
        "- You must be supportive, concise, and educational.\n"
        "- If the student asks for a re-grade, explain that this chat cannot change marks and they should use the formal challenge or teacher review process.\n"
    )

    context_payload = {
        "question": req.question.dict(),
        "student_response": req.student_response.dict(),
        "grade_result": req.grade_result,
    }

    user_text = (
        "You are helping a student understand their SAQ feedback.\n\n"
        f"Context:\n{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
        "Conversation so far:\n"
    )

    for msg in req.chat_history[-12:]:
        user_text += f"{msg.role.upper()}: {msg.content}\n"

    user_text += f"\nSTUDENT'S NEW MESSAGE:\n{req.user_message}\n\n"
    user_text += (
        "Reply as Vox-LM. Keep the answer educational and clear. "
        "Do not alter the grade."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    debug_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_text}"
    return messages, debug_prompt

#helper to build messages for answer refinement assistant
def build_refine_model_answer_messages(
    req: RefineModelAnswerRequest,
) -> tuple[List[Dict[str, Any]], str]:
    system_prompt = (
        f"You are Vox-LM, an expert assessment-design assistant for {req.discipline}. "
        "You help teachers improve model answers and rubrics so they are easier to use for fair SAQ marking.\n\n"
        "Return only a single valid JSON object and nothing else.\n"
    )

    rubric_summary = build_rubric_summary(req.question)

    user_text = (
        f"{rubric_summary}\n\n"
        "Task:\n"
        "Evaluate the current model answer and marking information for clarity, structure, and usefulness for AI-assisted SAQ marking.\n\n"
        "Assess whether the model answer:\n"
        "- Clearly identifies required key points.\n"
        "- Separates essential from optional points.\n"
        "- Supports partial-credit marking.\n"
        "- Avoids ambiguity.\n"
        "- Is structured enough for consistent marking.\n"
        "- Aligns with the question and total score.\n\n"
        "Then rewrite the model answer into a clearer marking-friendly structure.\n\n"
        "Return this exact JSON structure:\n"
        "{\n"
        '  "rating_score": number,\n'
        '  "rating_label": string,\n'
        '  "strengths": [string, ...],\n'
        '  "issues": [string, ...],\n'
        '  "rewritten_model_answer": string,\n'
        '  "suggested_marking_structure": string\n'
        "}\n\n"
        "rating_score must be from 0 to 100.\n"
        "rating_label should be one of: Poor, Fair, Good, Very good, Excellent.\n"
        "Do not include markdown outside the JSON."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    debug_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_text}"
    return messages, debug_prompt

#JSON Extraction
def extract_json(text: str) -> dict:
    """
    Try to extract a JSON object from the model output.

    Strategy:
    1. If there's a ```json ... ``` fenced block, use its contents.
    2. Else, take the first {...} block.
    3. Try json.loads directly.
    4. If that fails, attempt a simple fix for trailing commas before } or ].
    """
    original_text = text


    fenced = re.search(r"```json(.*?)```", text, flags=re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    else:

        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            logger.error("No JSON object found in model output:\n%s", text)
            raise ValueError("No JSON object found in model output")
        candidate = m.group(0).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e1:

        fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            logger.error(
                "Failed to parse JSON after cleanup.\nOriginal output:\n%s\n\nCandidate JSON:\n%s\n\nError1: %s\nError2: %s",
                original_text,
                candidate,
                e1,
                e2,
            )
            raise ValueError(f"Failed to parse model JSON: {e2}")

# Video MCQ Generator Helpers
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  #I would use int8 for CPU

_whisper_model = None


def get_whisper_model():
    """
    Lazy-load Whisper so your normal SAQ app does not pay startup cost unless video MCQ is used.
    """
    global _whisper_model

    if _whisper_model is None:
        logger.info(
            "Loading faster-whisper model: size=%s device=%s compute_type=%s",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    return _whisper_model


def call_video_llm(messages: List[Dict[str, Any]], max_new_tokens: int = 1800) -> str:
    """
    Adapter for the video MCQ module.

    Currently uses your existing Qwen3-VL call. Later, if you run Gemma E4 in
    a separate service, replace this function only.
    """
    return call_qwen_vl(messages, max_new_tokens=max_new_tokens)


def save_upload_to_tempfile(upload_file: UploadFile) -> str:
    suffix = Path(upload_file.filename or "uploaded_video.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = upload_file.file.read()
        tmp.write(data)
        return tmp.name


def get_video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()

    if fps <= 0:
        return 0.0

    return float(frame_count / fps)


def extract_audio_to_wav(video_path: str) -> str:
    """
    Extract mono 16 kHz WAV audio for Whisper.
    """
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:1000]}")

    return wav_path


def transcribe_audio_with_timestamps(audio_path: str) -> List[Dict[str, Any]]:
    """
    Returns transcript segments:
    [
      {"start": 0.0, "end": 4.2, "text": "..."},
      ...
    ]
    """
    whisper = get_whisper_model()

    segments, info = whisper.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )

    rows = []

    for seg in segments:
        text = str(seg.text or "").strip()
        if not text:
            continue

        rows.append(
            {
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": text,
            }
        )

    return rows


def extract_video_frames(
    video_path: str,
    interval_seconds: int = 10,
    max_frames: int = 100,
) -> List[Dict[str, Any]]:
    """
    Extract frames every N seconds.

    Returns:
    [
      {
        "timestamp_seconds": 30.0,
        "image": PIL.Image
      },
      ...
    ]
    """
    frames = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    duration = get_video_duration_seconds(video_path)
    if duration <= 0:
        cap.release()
        return frames

    timestamps = []
    t = 0.0

    while t <= duration:
        timestamps.append(t)
        t += float(interval_seconds)

    if len(timestamps) > max_frames:
        # Evenly sample if too many frames.
        idxs = np.linspace(0, len(timestamps) - 1, max_frames).astype(int)
        timestamps = [timestamps[i] for i in idxs]

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame_bgr = cap.read()

        if not ok or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).convert("RGB")
        img = resize_image_for_vlm(img, max_side=1280)

        frames.append(
            {
                "timestamp_seconds": round(float(ts), 2),
                "image": img,
            }
        )

    cap.release()
    return frames


def summarise_single_frame_with_vlm(
    img: Image.Image,
    timestamp_seconds: float,
    discipline: str,
) -> Dict[str, Any]:
    """
    Uses the VLM to OCR/summarise a single frame.
    """
    system_prompt = (
        f"You are an expert educational video analysis assistant for {discipline}. "
        "You analyse individual video frames from teaching videos. "
        "Return only valid JSON."
    )

    user_text = (
        "Analyse this video frame from a teaching video.\n\n"
        "Tasks:\n"
        "1. Identify any visible slide title or major heading.\n"
        "2. Extract important visible text using OCR where possible.\n"
        "3. Describe diagrams, clinical images, charts, or whiteboard content if present.\n"
        "4. Explain why this frame may be educationally relevant.\n\n"
        f"Frame timestamp: {timestamp_seconds} seconds.\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  "timestamp_seconds": number,\n'
        '  "visible_title": string,\n'
        '  "ocr_text": string,\n'
        '  "visual_summary": string,\n'
        '  "educational_relevance": string,\n'
        '  "confidence": number\n'
        "}\n"
        "Do not include text before or after the JSON."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    try:
        raw = call_video_llm(messages, max_new_tokens=700)
        pred = extract_json(raw)
    except Exception as e:
        pred = {
            "timestamp_seconds": timestamp_seconds,
            "visible_title": "",
            "ocr_text": "",
            "visual_summary": "",
            "educational_relevance": "",
            "confidence": 0.0,
            "error": str(e),
        }

    pred.setdefault("timestamp_seconds", timestamp_seconds)
    pred.setdefault("visible_title", "")
    pred.setdefault("ocr_text", "")
    pred.setdefault("visual_summary", "")
    pred.setdefault("educational_relevance", "")
    pred.setdefault("confidence", 50.0)

    try:
        pred["confidence"] = max(0.0, min(100.0, float(pred["confidence"])))
    except Exception:
        pred["confidence"] = 50.0

    return pred


def summarise_video_frames(
    frames: List[Dict[str, Any]],
    discipline: str,
) -> List[Dict[str, Any]]:
    summaries = []

    for frame in frames:
        img = frame["image"]
        ts = frame["timestamp_seconds"]

        summary = summarise_single_frame_with_vlm(
            img=img,
            timestamp_seconds=ts,
            discipline=discipline,
        )
        summaries.append(summary)

    return summaries


def chunk_transcript_segments(
    transcript_segments: List[Dict[str, Any]],
    chunk_seconds: int = 75,
) -> List[Dict[str, Any]]:
    """
    Merge Whisper segments into larger teaching chunks.
    """
    if not transcript_segments:
        return []

    chunks = []
    current = {
        "start": transcript_segments[0]["start"],
        "end": transcript_segments[0]["end"],
        "text_parts": [],
    }

    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = str(seg.get("text", "")).strip()

        if not seg_text:
            continue

        if seg_start - float(current["start"]) <= chunk_seconds:
            current["end"] = seg_end
            current["text_parts"].append(seg_text)
        else:
            chunks.append(
                {
                    "start": round(float(current["start"]), 2),
                    "end": round(float(current["end"]), 2),
                    "text": " ".join(current["text_parts"]).strip(),
                }
            )
            current = {
                "start": seg_start,
                "end": seg_end,
                "text_parts": [seg_text],
            }

    if current["text_parts"]:
        chunks.append(
            {
                "start": round(float(current["start"]), 2),
                "end": round(float(current["end"]), 2),
                "text": " ".join(current["text_parts"]).strip(),
            }
        )

    return chunks


def attach_nearby_frame_summaries_to_chunks(
    transcript_chunks: List[Dict[str, Any]],
    frame_summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Attach frame summaries whose timestamps fall inside or near each transcript chunk.
    """
    enriched = []

    for idx, chunk in enumerate(transcript_chunks):
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))

        nearby_frames = []

        for fs in frame_summaries:
            ts = float(fs.get("timestamp_seconds", -9999))
            if start - 5 <= ts <= end + 5:
                nearby_frames.append(
                    {
                        "timestamp_seconds": fs.get("timestamp_seconds"),
                        "visible_title": fs.get("visible_title", ""),
                        "ocr_text": truncate_text(fs.get("ocr_text", ""), 700),
                        "visual_summary": truncate_text(fs.get("visual_summary", ""), 500),
                        "educational_relevance": truncate_text(fs.get("educational_relevance", ""), 400),
                    }
                )

        enriched.append(
            {
                "segment_id": f"seg_{idx+1:03d}",
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "transcript": truncate_text(chunk.get("text", ""), 1800),
                "nearby_frame_summaries": nearby_frames,
            }
        )

    return enriched

def resize_image_for_vlm(img: Image.Image, max_side: int = 1280) -> Image.Image:
    w, h = img.size
    largest = max(w, h)

    if largest <= max_side:
        return img

    scale = max_side / float(largest)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

def build_video_mcq_generation_messages(
    discipline: str,
    topic: str,
    target_level: str,
    learning_objectives: List[str],
    duration_seconds: float,
    enriched_segments: List[Dict[str, Any]],
    num_check_questions: int,
    allow_anticipatory: bool,
    candidate_anticipatory_moments: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Build the final MCQ generation prompt.
    """
    candidate_anticipatory_moments = candidate_anticipatory_moments or []
    system_prompt = (
        f"You are Vox-LM, an expert educational assessment designer for {discipline}. "
        "You create high-quality multiple-choice questions from teaching videos. "
        "You must return only a single valid JSON object and nothing else.\n\n"
        "CRITICAL FORMAT INSTRUCTIONS:\n"
        "- Return one valid JSON object only.\n"
        "- Use double quotes only.\n"
        "- No markdown outside JSON.\n"
        "- No trailing commas.\n"
    )

    metadata = {
        "discipline": discipline,
        "topic": topic,
        "target_level": target_level,
        "teacher_learning_objectives": learning_objectives,
        "duration_seconds": round(float(duration_seconds), 2),
        "requested_number_of_in_video_questions": int(num_check_questions),
        "allow_anticipatory_questions": bool(allow_anticipatory),
        "candidate_anticipatory_moments_available": len(candidate_anticipatory_moments),
    }

    user_text = (
        "You are given a timestamped representation of a teaching video. "
        "The representation includes transcript segments and visual/frame summaries from slides or video frames.\n\n"

        "Your task is to create a teacher-reviewable set of MCQs for an interactive video activity.\n\n"
        "The JSON field is called embedded_questions for compatibility, but it should contain all in-video questions. "
        "Each item in embedded_questions may have question_type either 'anticipatory' or 'embedded_check'.\n\n"

        "QUESTION TYPES:\n"

        "1. pre_question:\n"
        "- Appears BEFORE the learner watches the video, at timestamp 0.\n"
        "- It must assess prior knowledge, clinical reasoning, prediction, or a likely misconception.\n"
        "- The pre_question should prepare the learner to notice the teaching point later, not tell them the teaching point now.\n"
        "- It must NOT be a direct recall question about the video's main teaching point.\n"
        "- It must NOT reveal the video's central conclusion too directly.\n"
        "- It must NOT ask 'which is the key factor', 'what does the video say', or similar recall-style wording.\n"
        "- It must NOT mention the video, lecture, transcript, slide, teacher, or what is shown/discussed later.\n"
        "- It must NOT use phrases such as 'according to the video', 'in the video', 'as shown', 'as discussed', 'as mentioned', or 'based on the lecture'.\n"
        "- Prefer a short clinical/practical scenario where the learner must reason from existing knowledge.\n"
        "- The stem should usually start with a real-world setup such as 'A patient...', 'A clinician...', 'Before treatment planning...', or 'When deciding...'.\n"
        "- The correct answer may prepare learners for a concept taught later, but the stem must not disclose that it is the video's key answer.\n"
        "- The pre-question should expose a misconception or create curiosity, not summarise the answer the video will teach.\n"
        "- The timestamp_seconds for pre_question must be 0.\n\n"
        "PRE-QUESTION STYLE EXAMPLES (FOR DENTISTRY TO HELP YOU, FOLLOW SIMILAR CONCEPT BUT NOT SCENARIOS FOR MEDICINE, LAW, EDUCATION):\n"
        "Bad pre-question because it reveals/retrieves the video's answer:\n"
        "- 'According to the video, which factor most influences the survival of resin-bonded bridges?'\n"
        "- 'Which design has the best survival rate according to the lecture?'\n"
        "- 'What is the key factor discussed in the video for RBB success?'\n"
        "- 'Which of the following is the main recommendation made in the video?'\n"
        "- 'What does the teacher say is most important for preventing failure?'\n\n"

        "Better pre-question because it checks prior reasoning/misconception without revealing the teaching point:\n"
        "- 'A patient is considering a minimally invasive replacement for a missing anterior tooth. Which planning factor is most likely to reduce the risk of failure?'\n"
        "- 'Before choosing a resin-bonded bridge design, which clinical issue should the dentist consider first?'\n"
        "- 'A learner assumes resin-bonded bridges mainly fail because of pontic shade mismatch. Which alternative factor is more clinically important?'\n"
        "- 'When planning a conservative bridge, which issue is most likely to affect whether the restoration remains stable over time?'\n"
        "- 'A clinician is choosing between treatment designs for a missing anterior tooth. What should they consider to avoid an avoidable failure?'\n\n"

        "The pre_question must use exactly one of these pre_question_strategy values:\n"
        "- misconception_probe\n"
        "- clinical_prediction\n"
        "- prerequisite_knowledge_check\n"
        "- curiosity_prompt\n\n"
        "Do not use a direct_recall strategy for the pre_question.\n"
        


        "2. embedded_check:\n"
        "- Appears during the video after a key explanation.\n"
        "- Checks understanding of recently explained content.\n"
        "- Should be placed at a meaningful timestamp, not randomly.\n\n"

        "3. anticipatory:\n"
        "- Appears before the teacher reveals or explains an answer.\n"
        "- Asks the learner to predict what comes next.\n"
        "- Use when the transcript suggests a suitable prediction moment, especially when the teacher asks a question, pauses, sets up a problem, or says something that is answered shortly afterwards.\n"
        "- Include reveal_after_seconds when possible.\n\n"

        "PRIORITY RULE FOR ANTICIPATORY QUESTIONS:\n"
        "- If allow_anticipatory_questions is true, you MUST first scan the transcript for teacher questions, prompts, pauses, or prediction moments.\n"
        "- If the teacher asks a clear question and then explains or reveals the answer within the next 10-90 seconds, create an anticipatory question at the time of the teacher's question.\n"
        "- Do NOT convert these moments into embedded_check questions unless there is no clear upcoming reveal.\n"
        "- If suitable teacher-question moments exist, generate anticipatory questions before filling remaining slots with embedded_check questions.\n"
        "- If allow_anticipatory_questions is true and there are multiple suitable teacher-question moments, at least half of the in-video questions should be anticipatory where possible.\n"
        "- Only use embedded_check questions for content that has already been explained and is being checked retrospectively.\n\n"

        "MCQ QUALITY RULES:\n"
        "- Each MCQ must have exactly four options: A, B, C, D.\n"
        "- There must be one clearly best answer.\n"
        "- Distractors must be plausible and based on likely misconceptions or confusions.\n"
        "- Avoid trick questions.\n"
        "- Avoid using 'all of the above' and 'none of the above'.\n"
        "- Avoid options where the correct answer is obviously longer or more detailed than the distractors.\n"
        "- Use clear student-friendly wording.\n"
        "- Each question must include feedback for correct and incorrect answers.\n"
        "- Each question must include a rationale for the teacher.\n"
        "- Each embedded question must include evidence_start_seconds and evidence_end_seconds.\n"
        "- If a question is uncertain or needs teacher checking, set teacher_review_recommended to true.\n\n"
        "- For pre_question only, never use video-dependent wording such as 'according to the video', 'in the video', 'as shown in the video', 'as discussed in the lecture', or 'based on the transcript'.\n"
        "- A pre_question should feel answerable before watching; embedded_questions may refer to recently explained video content or anticipatory questions, but pre_question must not.\n"

        "SOURCE RULES FOR PRE_QUESTION:\n"
        "- When creating the pre_question, use ONLY the broad topic, target level, discipline, and teacher learning objectives.\n"
        "- Do NOT use the timestamped transcript evidence as the direct source for the pre_question stem.\n"
        "- Do NOT turn a key teaching point from the video into a pre_question.\n"
        "- Do NOT copy or paraphrase a conclusion that appears later in the video as the pre_question answer.\n"
        "- The pre_question should check prerequisite reasoning, a common misconception, or a prediction that prepares the learner for the video.\n"
        "- The pre_question should make the learner curious about the video, not reveal the video's answer in advance.\n"
        "- If the obvious question would be 'what is the key factor?', rewrite it as a clinical scenario (for medicine or dentistry only), misconception probe, or planning decision.\n"
        "- The correct option should be clinically reasonable, but the stem should not announce that this is the video's main conclusion.\n\n"


        f"Video metadata:\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n\n"

        f"Timestamped video evidence:\n{json.dumps(enriched_segments, indent=2, ensure_ascii=False)}\n\n"

        f"Candidate anticipatory moments:\n{json.dumps(candidate_anticipatory_moments, indent=2, ensure_ascii=False)}\n\n"

        "STRICT ANTICIPATORY REQUIREMENT:\n"
        "- If allow_anticipatory_questions is true and Candidate anticipatory moments are provided, you MUST use at least one of them to create an anticipatory question.\n"
        "- Use candidate anticipatory moments only when the teacher_prompt_or_question is a genuine learner-facing question, prompt, or prediction opportunity.\n"
        "- Do not use introductory statements such as 'what we are going to do' or 'today we will...' as anticipatory moments unless they contain a clear learner-facing question.\n"
        "- Prefer candidates with higher quality_score where available.\n"
        "- The anticipatory question timestamp_seconds must match the candidate's question_timestamp_seconds.\n"
        "- The anticipatory question should ask the learner to predict the answer before the upcoming explanation.\n"
        "- The reveal_after_seconds should point approximately from the question timestamp to the later reveal/explanation.\n"
        "- Do not label these candidate-based questions as embedded_check.\n"
        "- Only use embedded_check for retrospective checks after content has already been explained.\n\n"


        "Return exactly this JSON structure:\n"
        "{\n"
        '  "transcript_summary": string,\n'
        '  "learning_objectives_detected": [string, ...],\n'
        '  "teaching_segments": [\n'
        '    {\n'
        '      "segment_id": string,\n'
        '      "start_seconds": number,\n'
        '      "end_seconds": number,\n'
        '      "summary": string,\n'
        '      "key_concepts": [string, ...],\n'
        '      "suitable_question_opportunity": boolean,\n'
        '      "suggested_question_type": string\n'
        '    }\n'
        '  ],\n'
        '  "pre_question": {\n'
        '    "question_id": "pre_001",\n'
        '    "question_type": "pre_question",\n'
        '    "pre_question_strategy": string,\n'
        '    "timestamp_seconds": 0,\n'
        '    "stem": string,\n'
        '    "options": [\n'
        '      {"label": "A", "text": string},\n'
        '      {"label": "B", "text": string},\n'
        '      {"label": "C", "text": string},\n'
        '      {"label": "D", "text": string}\n'
        '    ],\n'
        '    "correct_option": string,\n'
        '    "feedback_correct": string,\n'
        '    "feedback_incorrect": string,\n'
        '    "rationale": string,\n'
        '    "learning_objective": string,\n'
        '    "difficulty": string,\n'
        '    "quality_flags": {\n'
        '      "single_best_answer": boolean,\n'
        '      "distractors_plausible": boolean,\n'
        '      "aligned_to_video": boolean,\n'
        '      "teacher_review_recommended": boolean\n'
        '    }\n'
        '  },\n'
        '  "embedded_questions": [\n'
        '    {\n'
        '      "question_id": string,\n'
        '      "question_type": string,\n'
        '      "timestamp_seconds": number,\n'
        '      "reveal_after_seconds": number,\n'
        '      "stem": string,\n'
        '      "options": [\n'
        '        {"label": "A", "text": string},\n'
        '        {"label": "B", "text": string},\n'
        '        {"label": "C", "text": string},\n'
        '        {"label": "D", "text": string}\n'
        '      ],\n'
        '      "correct_option": string,\n'
        '      "feedback_correct": string,\n'
        '      "feedback_incorrect": string,\n'
        '      "rationale": string,\n'
        '      "learning_objective": string,\n'
        '      "difficulty": string,\n'
        '      "evidence_start_seconds": number,\n'
        '      "evidence_end_seconds": number,\n'
        '      "placement_reason": string,\n'
        '      "quality_flags": {\n'
        '        "single_best_answer": boolean,\n'
        '        "distractors_plausible": boolean,\n'
        '        "aligned_to_video": boolean,\n'
        '        "teacher_review_recommended": boolean\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        "Important:\n"
        f"- Generate exactly 1 pre_question and up to {int(num_check_questions)} in-video questions inside the embedded_questions array.\n"
        "- Each in-video question must have question_type either 'anticipatory' or 'embedded_check'.\n"
       "- If allow_anticipatory_questions is true and candidate anticipatory moments are provided, at least one embedded_questions item MUST have question_type = \"anticipatory\".\n"
        "- If multiple high-quality candidate anticipatory moments are available, use anticipatory questions before filling remaining slots with embedded_check questions.\n"
        "- The pre_question must be a prior-knowledge, misconception, prediction, or clinical reasoning question.\n"
        "- The pre_question must be generated from the broad topic/objectives, not from a specific later transcript statement.\n"
        "- The pre_question must not directly reveal the video's main conclusion or key takeaway.\n"
        "- The pre_question must not ask the learner to identify the video's key factor, main message, preferred design, or final recommendation.\n"
        "- The pre_question must not be phrased as a recall question about the video.\n"
        "- The pre_question must not mention the video, lecture, transcript, slide, teacher, or any phrase like 'according to the video'.\n"
        "- The pre_question stem should normally be scenario-based, misconception-based, or prediction-based.\n"
        "- The pre_question should prepare the learner to notice the teaching point later, not tell them the teaching point now.\n"
        "- Embedded questions may check content recently explained in the video.\n"
        "- If anticipatory questions are not allowed, use only embedded_check for embedded_questions.\n"
        "- Before returning the JSON, silently check the pre_question stem.\n"
        "- If the pre_question sounds like it was written after watching the video, rewrite it before outputting JSON.\n"
        "- If the pre_question reveals the video's main answer too directly, rewrite it as a scenario or misconception probe before outputting JSON.\n"
        "- If the pre_question uses 'according to the video', 'in the video', 'as discussed', 'key factor discussed', or similar wording, rewrite it before outputting JSON.\n"

        "- Do not include text before or after the JSON."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    debug_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_text}"
    return messages, debug_prompt

def normalise_mcq_question_type(raw_type: Any, fallback_type: str, item: Dict[str, Any]) -> str:
    qtype = str(raw_type or "").strip().lower()

    if qtype in ["pre_question", "pre-question", "pre"]:
        return "pre_question"

    if qtype in ["anticipatory", "prediction", "prediction_question", "predictive"]:
        return "anticipatory"

    if qtype in ["embedded_check", "embedded", "check", "embedded_question", "check_in", "check-in"]:
        return "embedded_check"

    # Infer anticipatory if the model supplied reveal_after_seconds.
    reveal = item.get("reveal_after_seconds", None)
    if reveal not in ["", None]:
        try:
            if float(reveal) > 0:
                return "anticipatory"
        except Exception:
            pass

    return fallback_type

def normalise_mcq_item(item: Dict[str, Any], fallback_id: str, fallback_type: str) -> Dict[str, Any]:
    item = item or {}

    item.setdefault("question_id", fallback_id)
    item["question_type"] = normalise_mcq_question_type(
        item.get("question_type", ""),
        fallback_type=fallback_type,
        item=item)
    item.setdefault("timestamp_seconds", 0)
    item.setdefault("stem", "")
    item.setdefault("options", [])
    item.setdefault("correct_option", "")
    item.setdefault("feedback_correct", "")
    item.setdefault("feedback_incorrect", "")
    item.setdefault("rationale", "")
    item.setdefault("learning_objective", "")
    item.setdefault("difficulty", "medium")
    item.setdefault("quality_flags", {})

    labels = ["A", "B", "C", "D"]
    options = item.get("options", [])

    clean_options = []
    if isinstance(options, list):
        for i, opt in enumerate(options[:4]):
            if isinstance(opt, dict):
                label = str(opt.get("label", labels[i] if i < len(labels) else "")).strip() or labels[i]
                text = str(opt.get("text", "")).strip()
            else:
                label = labels[i] if i < len(labels) else str(i + 1)
                text = str(opt).strip()

            clean_options.append({"label": label, "text": text})

    while len(clean_options) < 4:
        clean_options.append({"label": labels[len(clean_options)], "text": ""})

    item["options"] = clean_options

    qf = item.get("quality_flags", {})
    if not isinstance(qf, dict):
        qf = {}

    qf.setdefault("single_best_answer", False)
    qf.setdefault("distractors_plausible", False)
    qf.setdefault("aligned_to_video", False)
    qf.setdefault("teacher_review_recommended", True)

    item["quality_flags"] = qf

    return item


def normalise_video_mcq_prediction(pred: Dict[str, Any], num_check_questions: int) -> Dict[str, Any]:
    pred = pred or {}

    pred.setdefault("transcript_summary", "")
    pred.setdefault("learning_objectives_detected", [])
    pred.setdefault("teaching_segments", [])
    pred.setdefault("pre_question", {})
    pred.setdefault("embedded_questions", [])

    pred["pre_question"] = normalise_mcq_item(
        pred.get("pre_question", {}),
        fallback_id="pre_001",
        fallback_type="pre_question",
    )

    embedded = pred.get("embedded_questions", [])
    if not isinstance(embedded, list):
        embedded = []

    clean_embedded = []
    for i, item in enumerate(embedded[:num_check_questions]):
        clean_embedded.append(
            normalise_mcq_item(
                item,
                fallback_id=f"check_{i+1:03d}",
                fallback_type="embedded_check",
            )
        )

    pred["embedded_questions"] = clean_embedded

    if not isinstance(pred.get("teaching_segments"), list):
        pred["teaching_segments"] = []

    return pred


def truncate_text(value: Any, max_len: int = 350) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...[truncated]"

def build_visual_only_segments_from_frames(
    frame_summaries: List[Dict[str, Any]],
    chunk_seconds: int = 75,
) -> List[Dict[str, Any]]:
    """
    Build evidence segments from frames alone.
    Useful when there is little/no transcript, or when slide frames fall outside transcript chunks.
    """
    if not frame_summaries:
        return []

    bins: Dict[int, List[Dict[str, Any]]] = {}

    for fs in frame_summaries:
        try:
            ts = float(fs.get("timestamp_seconds", 0.0))
        except Exception:
            ts = 0.0

        bin_id = int(ts // chunk_seconds)
        bins.setdefault(bin_id, []).append(fs)

    visual_segments = []

    for idx, bin_id in enumerate(sorted(bins.keys()), start=1):
        items = bins[bin_id]
        timestamps = []

        nearby_frames = []

        for fs in items:
            try:
                ts = float(fs.get("timestamp_seconds", 0.0))
            except Exception:
                ts = 0.0

            timestamps.append(ts)

            nearby_frames.append(
                {
                    "timestamp_seconds": fs.get("timestamp_seconds"),
                    "visible_title": fs.get("visible_title", ""),
                    "ocr_text": truncate_text(fs.get("ocr_text", ""), 900),
                    "visual_summary": truncate_text(fs.get("visual_summary", ""), 700),
                    "educational_relevance": truncate_text(fs.get("educational_relevance", ""), 500),
                }
            )

        start = min(timestamps) if timestamps else float(bin_id * chunk_seconds)
        end = max(timestamps) if timestamps else float((bin_id + 1) * chunk_seconds)

        visual_segments.append(
            {
                "segment_id": f"vis_{idx:03d}",
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "transcript": "",
                "nearby_frame_summaries": nearby_frames,
            }
        )

    return visual_segments

def extract_candidate_anticipatory_moments(
    transcript_segments: List[Dict[str, Any]],
    min_reveal_delay: float = 5.0,
    max_reveal_delay: float = 90.0,
    max_candidates: int = 8,
    min_question_time_seconds: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Identify likely anticipatory-question moments from raw Whisper segments.

    Stricter version:
    - Prioritises genuine learner-facing questions.
    - Avoids broad false positives from ordinary words like "what" or "how".
    - Avoids very early introductory statements unless they contain a real question mark.
    """

    if not transcript_segments:
        return []

    strong_question_patterns = [
        r"\?",
        r"^\s*(what|why|how|which|when|where)\s+(should|would|could|can|do|does|did|is|are|was|were|will|might)\b",
        r"\bwhat do you think\b",
        r"\bcan you\b",
        r"\bdo you think\b",
        r"\bwould you\b",
        r"\bshould we\b",
        r"\bpredict\b",
        r"\bguess\b",
        r"\bbefore I tell you\b",
        r"\blet'?s think\b",
        r"\bthe question is\b",
    ]

    weak_intro_or_explanation_patterns = [
        r"\bwhat we'?re going to\b",
        r"\bwhat we are going to\b",
        r"\bwhat I'?m going to\b",
        r"\bwhat I am going to\b",
        r"\bwhat I want to\b",
        r"\bwhat you can see\b",
        r"\bwhat we can see\b",
        r"\bwhat happens here\b",
        r"\bwhat this shows\b",
        r"\bhow we do this\b",
        r"\bhow this works\b",
        r"\bwe are going to\b",
        r"\btoday we\b",
        r"\bin this video\b",
        r"\bin this lecture\b",
    ]

    candidates: List[Dict[str, Any]] = []

    for i, seg in enumerate(transcript_segments):
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue

        lower = text.lower()

        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except Exception:
            start = 0.0
            end = start

        has_question_mark = "?" in text

        # Avoid very early intro moments unless the transcript has a real question mark.
        if start < min_question_time_seconds and not has_question_mark:
            continue

        has_strong_question = any(
            re.search(pattern, lower, flags=re.I)
            for pattern in strong_question_patterns
        )

        if not has_strong_question:
            continue

        # Filter out explanatory/introductory "what/how" phrases.
        looks_like_intro = any(
            re.search(pattern, lower, flags=re.I)
            for pattern in weak_intro_or_explanation_patterns
        )

        if looks_like_intro and not has_question_mark:
            continue

        reveal_start = end + min_reveal_delay
        reveal_end = end + max_reveal_delay

        upcoming_parts = []

        for future in transcript_segments[i + 1:]:
            try:
                f_start = float(future.get("start", 0.0))
            except Exception:
                continue

            if reveal_start <= f_start <= reveal_end:
                future_text = str(future.get("text", "") or "").strip()
                if future_text:
                    upcoming_parts.append(future_text)

            if f_start > reveal_end:
                break

        upcoming_text = " ".join(upcoming_parts).strip()

        if not upcoming_text:
            continue

        # Basic quality score.
        score = 0
        if has_question_mark:
            score += 3
        if re.search(r"\bwhat do you think\b|\bcan you\b|\bdo you think\b|\bpredict\b|\bguess\b", lower, flags=re.I):
            score += 2
        if start >= min_question_time_seconds:
            score += 1
        if len(upcoming_text) >= 120:
            score += 1

        candidates.append(
            {
                "candidate_id": f"anticipatory_candidate_{len(candidates) + 1:03d}",
                "question_timestamp_seconds": round(start, 2),
                "teacher_prompt_or_question": truncate_text(text, 300),
                "possible_reveal_window_seconds": [
                    round(reveal_start, 2),
                    round(reveal_end, 2),
                ],
                "upcoming_explanation_excerpt": truncate_text(upcoming_text, 600),
                "quality_score": score,
            }
        )

    # Prefer higher-quality candidates, then earlier candidates.
    candidates = sorted(
        candidates,
        key=lambda x: (
            -int(x.get("quality_score", 0)),
            float(x.get("question_timestamp_seconds", 0.0)),
        ),
    )

    return candidates[:max_candidates]



def count_anticipatory_questions(pred: Dict[str, Any]) -> int:
    embedded = pred.get("embedded_questions", []) or []
    count = 0

    for q in embedded:
        qtype = str(q.get("question_type", "") or "").strip().lower()
        if qtype in ["anticipatory", "prediction", "prediction_question"]:
            count += 1

    return count


def transcript_has_teacher_question_cues(enriched_segments: List[Dict[str, Any]]) -> bool:
    cue_patterns = [
        r"\?",
        r"\bwhat\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bwhich\b",
        r"\bwhen\b",
        r"\bwhere\b",
        r"\bcan you\b",
        r"\bdo you think\b",
        r"\bwhat do you think\b",
        r"\bpredict\b",
        r"\bguess\b",
        r"\bbefore I tell you\b",
        r"\blet's think\b",
        r"\bconsider\b",
    ]

    combined_text = []

    for seg in enriched_segments or []:
        combined_text.append(str(seg.get("transcript", "") or ""))

    text = " ".join(combined_text).lower()

    if not text.strip():
        return False

    return any(re.search(pattern, text, flags=re.I) for pattern in cue_patterns)


def build_enriched_video_evidence(
    transcript_segments: List[Dict[str, Any]],
    frame_summaries: List[Dict[str, Any]],
    chunk_seconds: int = 30,
) -> List[Dict[str, Any]]:
    """
    Combine transcript chunks and frame summaries.
    Falls back to visual-only evidence if transcript is absent.
    Also appends orphan frame summaries that were not attached to transcript chunks.
    """
    transcript_chunks = chunk_transcript_segments(
        transcript_segments=transcript_segments,
        chunk_seconds=chunk_seconds,
    )

    if not transcript_chunks:
        return build_visual_only_segments_from_frames(
            frame_summaries=frame_summaries,
            chunk_seconds=chunk_seconds,
        )

    enriched = attach_nearby_frame_summaries_to_chunks(
        transcript_chunks=transcript_chunks,
        frame_summaries=frame_summaries,
    )

    attached_timestamps = set()

    for seg in enriched:
        for fs in seg.get("nearby_frame_summaries", []) or []:
            try:
                attached_timestamps.add(round(float(fs.get("timestamp_seconds", 0.0)), 2))
            except Exception:
                pass

    orphan_frames = []

    for fs in frame_summaries:
        try:
            ts = round(float(fs.get("timestamp_seconds", 0.0)), 2)
        except Exception:
            ts = 0.0

        if ts not in attached_timestamps:
            orphan_frames.append(fs)

    if orphan_frames:
        visual_only_segments = build_visual_only_segments_from_frames(
            frame_summaries=orphan_frames,
            chunk_seconds=chunk_seconds,
        )
        enriched.extend(visual_only_segments)

    return enriched

def assign_performance_tiers(scores: pd.Series, max_score: Optional[float]) -> pd.Series:
    scores = pd.to_numeric(scores, errors="coerce")
    tiers = pd.Series(index=scores.index, dtype="object")

    if max_score is not None and float(max_score) > 0:
        pct = (scores / float(max_score)) * 100.0
        tiers.loc[pct < 50] = "low"
        tiers.loc[(pct >= 50) & (pct < 70)] = "mid"
        tiers.loc[pct >= 70] = "high"
    else:
        valid = scores.dropna()
        try:
            ranked = valid.rank(method="first")
            q = pd.qcut(ranked, q=3, labels=["low", "mid", "high"])
            tiers.loc[q.index] = q.astype(str)
        except Exception:
            median_val = valid.median() if not valid.empty else 0.0
            tiers.loc[scores < median_val] = "low"
            tiers.loc[scores == median_val] = "mid"
            tiers.loc[scores > median_val] = "high"

    return tiers.fillna("unscored")


def compute_score_distribution(scores: pd.Series, max_score: Optional[float]) -> Dict[str, int]:
    scores = pd.to_numeric(scores, errors="coerce").dropna()

    if scores.empty:
        return {
            "0-24%": 0,
            "25-49%": 0,
            "50-74%": 0,
            "75-100%": 0,
        }

    if max_score is not None and float(max_score) > 0:
        pct = (scores / float(max_score)) * 100.0
        labels = ["0-24%", "25-49%", "50-74%", "75-100%"]
        bands = pd.cut(
            pct,
            bins=[-0.01, 25, 50, 75, 100.01],
            labels=labels,
            include_lowest=True,
            right=False,
        )
        return {label: int((bands == label).sum()) for label in labels}

    # Fallback if no max_score exists
    return {
        "lower_third": int((scores <= scores.quantile(0.33)).sum()),
        "middle_third": int(((scores > scores.quantile(0.33)) & (scores < scores.quantile(0.67))).sum()),
        "upper_third": int((scores >= scores.quantile(0.67)).sum()),
    }


def compute_overall_subquestion_stats(
    scored_df: pd.DataFrame,
    question: Question,
) -> Dict[str, Any]:
    sub_stats: Dict[str, Any] = {}
    subquestions = question.subquestions or []

    sub_max_lookup = {}
    for sq in subquestions:
        sid = str(sq.get("id"))
        sub_max_lookup[f"sub_{sid}"] = sq.get("max_score")

    sub_cols = [c for c in scored_df.columns if str(c).startswith("sub_")]
    for col in sub_cols:
        vals = pd.to_numeric(scored_df[col], errors="coerce")
        mean_score = float(vals.mean()) if vals.notna().any() else 0.0
        sub_max = sub_max_lookup.get(str(col))

        if sub_max is not None:
            try:
                pct = (mean_score / float(sub_max)) * 100.0 if float(sub_max) > 0 else None
            except Exception:
                pct = None
        else:
            pct = None

        sub_stats[str(col)] = {
            "mean_score": round(mean_score, 2),
            "mean_percent_of_sub_max": round(pct, 1) if pct is not None else None,
        }

    return sub_stats


def compute_tier_subquestion_stats(
    scored_df: pd.DataFrame,
    question: Question,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for tier in ["high", "mid", "low"]:
        tier_df = scored_df[scored_df["performance_tier"] == tier]
        result[tier] = compute_overall_subquestion_stats(tier_df, question)
    return result


def select_representative_examples(
    tier_df: pd.DataFrame,
    tier_name: str,
    response_col: Optional[str],
    max_examples: int,
    max_score: Optional[float],
) -> List[Dict[str, Any]]:
    if tier_df.empty or max_examples <= 0:
        return []

    work = tier_df.copy()

    if tier_name == "low":
        chosen = work.sort_values("total_score", ascending=True).head(max_examples)
    elif tier_name == "high":
        chosen = work.sort_values("total_score", ascending=False).head(max_examples)
    else:
        med = float(work["total_score"].median()) if not work.empty else 0.0
        chosen = (
            work.assign(_dist=(work["total_score"] - med).abs())
            .sort_values(["_dist", "total_score"], ascending=[True, False])
            .head(max_examples)
        )

    sub_cols = [c for c in work.columns if str(c).startswith("sub_")]
    examples: List[Dict[str, Any]] = []

    for _, row in chosen.iterrows():
        student_id = ""
        if "STUDENT_ID" in row.index and pd.notna(row["STUDENT_ID"]):
            student_id = str(row["STUDENT_ID"])

        total_score = float(row["total_score"])
        score_percent = None
        if max_score is not None and float(max_score) > 0:
            score_percent = round((total_score / float(max_score)) * 100.0, 1)

        sub_scores = {}
        for col in sub_cols:
            val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            sub_scores[str(col)] = None if pd.isna(val) else float(val)

        examples.append(
            {
                "student_id": student_id,
                "total_score": round(total_score, 2),
                "score_percent": score_percent,
                "response_excerpt": truncate_text(row.get(response_col, "") if response_col else "", 300),
                "rationale": truncate_text(row.get("rationale", ""), 280),
                "feedback_overall": truncate_text(row.get("feedback_overall", ""), 220),
                "sub_scores": sub_scores,
            }
        )

    return examples

def build_local_tiered_summary_data(
    csv_text: str,
    question: Question,
    has_subquestions: bool,
    max_examples_per_tier: int,
) -> Dict[str, Any]:
    df = pd.read_csv(io.StringIO(csv_text))

    if "total_score" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'total_score' column.")

    total_students = int(len(df))
    numeric_scores = pd.to_numeric(df["total_score"], errors="coerce")

    scored_df = df.loc[numeric_scores.notna()].copy()
    scored_df["total_score"] = pd.to_numeric(scored_df["total_score"], errors="coerce")
    scored_students = int(len(scored_df))

    if scored_df.empty:
        return {
            "total_students": total_students,
            "scored_students": 0,
            "class_average": 0.0,
            "median_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "tier_thresholds": {
                "high": ">= 70%",
                "mid": "50% to < 70%",
                "low": "< 50%",
            },
            "tier_counts": {"high": 0, "mid": 0, "low": 0},
            "score_distribution": {"0-24%": 0, "25-49%": 0, "50-74%": 0, "75-100%": 0},
            "overall_subquestion_stats": {},
            "tier_subquestion_stats": {"high": {}, "mid": {}, "low": {}},
            "tier_examples": {"high": [], "mid": [], "low": []},
            "observed_misconception_patterns": [],
            "observed_out_of_scope_patterns": []
        }

    q_max = question.max_score if question.max_score is not None else None

    class_average = float(scored_df["total_score"].mean())
    median_score = float(scored_df["total_score"].median())
    max_score_observed = float(scored_df["total_score"].max())
    min_score_observed = float(scored_df["total_score"].min())
    std_score = float(scored_df["total_score"].std(ddof=0)) if scored_students > 1 else 0.0

    scored_df["performance_tier"] = assign_performance_tiers(scored_df["total_score"], q_max)

    tier_counts = {
        "high": int((scored_df["performance_tier"] == "high").sum()),
        "mid": int((scored_df["performance_tier"] == "mid").sum()),
        "low": int((scored_df["performance_tier"] == "low").sum()),
    }

    response_cols = [c for c in df.columns if str(c).startswith("STUDENT RESPONSES")]
    response_col = response_cols[0] if response_cols else None

    overall_sub_stats = compute_overall_subquestion_stats(scored_df, question) if has_subquestions else {}
    tier_sub_stats = compute_tier_subquestion_stats(scored_df, question) if has_subquestions else {
        "high": {}, "mid": {}, "low": {}
    }

    tier_examples = {
        "high": select_representative_examples(
            scored_df[scored_df["performance_tier"] == "high"],
            "high",
            response_col,
            max_examples_per_tier,
            q_max,
        ),
        "mid": select_representative_examples(
            scored_df[scored_df["performance_tier"] == "mid"],
            "mid",
            response_col,
            max_examples_per_tier,
            q_max,
        ),
        "low": select_representative_examples(
            scored_df[scored_df["performance_tier"] == "low"],
            "low",
            response_col,
            max_examples_per_tier,
            q_max,
        ),
    }

    observed_misconception_patterns = extract_highlight_pattern_counts(
    scored_df,
    category="misconception",
    max_items=12,
    )

    observed_out_of_scope_patterns = extract_highlight_pattern_counts(
        scored_df,
        category="out_of_scope",
        max_items=12,
    )

    return {
        "total_students": total_students,
        "scored_students": scored_students,
        "class_average": round(class_average, 2),
        "median_score": round(median_score, 2),
        "max_score": round(max_score_observed, 2),
        "min_score": round(min_score_observed, 2),
        "std_score": round(std_score, 2),
        "tier_thresholds": {
            "high": ">= 70%",
            "mid": "50% to < 70%",
            "low": "< 50%",
        },
        "tier_counts": tier_counts,
        "score_distribution": compute_score_distribution(scored_df["total_score"], q_max),
        "overall_subquestion_stats": overall_sub_stats,
        "tier_subquestion_stats": tier_sub_stats,
        "tier_examples": tier_examples,
        "observed_misconception_patterns": observed_misconception_patterns,
        "observed_out_of_scope_patterns": observed_out_of_scope_patterns
    }

def build_batch_summary_messages(
    csv_text: str,
    question: Question,
    has_subquestions: bool,
    discipline: str,
    max_examples_per_tier: int = 5,
):
    local_data = build_local_tiered_summary_data(
        csv_text=csv_text,
        question=question,
        has_subquestions=has_subquestions,
        max_examples_per_tier=max_examples_per_tier,
    )

    rubric_summary = build_rubric_summary(question)

    evidence_payload = {
        "exact_class_statistics": {
            "total_students": local_data["total_students"],
            "scored_students": local_data["scored_students"],
            "class_average": local_data["class_average"],
            "median_score": local_data["median_score"],
            "max_score": local_data["max_score"],
            "min_score": local_data["min_score"],
            "std_score": local_data["std_score"],
            "tier_thresholds": local_data["tier_thresholds"],
            "tier_counts": local_data["tier_counts"],
            "score_distribution": local_data["score_distribution"],
        },
        "overall_subquestion_stats": local_data["overall_subquestion_stats"],
        "tier_subquestion_stats": local_data["tier_subquestion_stats"],
        "representative_examples_by_tier": local_data["tier_examples"],
        "observed_misconception_patterns": local_data.get("observed_misconception_patterns", []),
        "observed_out_of_scope_patterns": local_data.get("observed_out_of_scope_patterns", []),
    }

    user_text = (
        f"{rubric_summary}\n\n"
        "You are given exact class statistics and representative examples grouped into high, mid, "
        "and low performers. These statistics were computed locally and are exact.\n\n"
        "Your task is to produce an interpretation for teachers that refers to the model answer, "
        "rubric, and subquestion expectations.\n\n"
        "Interpret the cohort by tier:\n"
        "- High performers: what they generally understood and what they did well.\n"
        "- Mid performers: what they partially understood, where their answers were incomplete, vague, or inconsistent.\n"
        "- Low performers: what they commonly misunderstood, omitted, or confused.\n\n"
        "Important instructions:\n"
        "- Use the rubric/model answer to explain likely strengths and misconceptions.\n"
        "- Base your conclusions only on the evidence provided below.\n"
        "- Do not recalculate the numbers; they are already exact.\n"
        "- Focus on patterns across the class, not isolated edge cases.\n"
        "- Keep recommendations practical for teachers.\n"
        "- For common_misconceptions and out_of_scope_points, use the exact observed pattern counts where available.\n"
        "- Include percent_students and student_count for each misconception or out-of-scope point.\n"
        "- If no reliable pattern is available, return an empty list for that field.\n"
        "- If subquestions exist, identify common errors by subquestion and give a short teaching note for each one.\n\n"
        f"Local evidence:\n{json.dumps(evidence_payload, indent=2)}\n\n"
        "Return a single valid JSON object with exactly this structure:\n"
        "{\n"
        '  "tier_summaries": {\n'
        '    "high": string,\n'
        '    "mid": string,\n'
        '    "low": string\n'
        "  },\n"
        '  "strengths": [string, ...],\n'
        '  "common_misconceptions": [\n'
        '    {\n'
        '      "point": string,\n'
        '      "percent_students": number,\n'
        '      "student_count": number\n'
        '    }\n'
        '  ],\n'
        '  "out_of_scope_points": [\n'
        '    {\n'
        '      "point": string,\n'
        '      "percent_students": number,\n'
        '      "student_count": number\n'
        '    }\n'
        '  ],\n'
        '  "weak_areas": [string, ...],\n'
        '  "teacher_next_steps": [string, ...],\n'
        '  "narrative_summary": string,\n'
        '  "subquestion_diagnostics": {\n'
        '    "<sub_id>": {\n'
        '      "common_errors": [string, ...],\n'
        '      "teaching_note": string\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "If there are no subquestions, return an empty object for subquestion_diagnostics.\n"
        "Do not include any text before or after the JSON."
    )


    system_prompt = get_summary_system_prompt(discipline)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    debug_prompt = (
        "SYSTEM PROMPT:\n"
        f"{system_prompt}\n\n"
        "USER PROMPT:\n"
        f"{user_text}"
    )

    return messages, debug_prompt, local_data

def build_solo_question_messages(question: Question, discipline: str):
    rubric_summary = build_rubric_summary(question)

    user_text = (
        f"{rubric_summary}\n\n"
        "Analyse this question using the SOLO taxonomy.\n"
        "Use only these SOLO labels:\n"
        "- prestructural\n"
        "- unistructural\n"
        "- multistructural\n"
        "- relational\n"
        "- extended_abstract\n\n"
        "Decide whether this question mainly assesses one SOLO level or multiple levels.\n"
        "If subquestions exist, map each subquestion to the main SOLO level it assesses.\n\n"
        "Return a single valid JSON object with exactly this structure:\n"
        "{\n"
        '  "question_targets_multiple_levels": boolean,\n'
        '  "overall_solo_levels": [string, ...],\n'
        '  "question_level_summary": string,\n'
        '  "subquestion_solo_map": {\n'
        '    "<sub_id>": string\n'
        "  }\n"
        "}\n"
    )

    system_prompt = get_solo_system_prompt(discipline)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    debug_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_text}"
    return messages, debug_prompt

# SOLO Analysis function
def analyse_question_solo(question: Question, discipline: str) -> tuple[Dict[str, Any], str]:
    messages, debug_prompt = build_solo_question_messages(question, discipline)

    try:
        raw_output = call_qwen_vl(messages, max_new_tokens=700)
        pred = extract_json(raw_output)
    except Exception:
        pred = {
            "question_targets_multiple_levels": bool(question.subquestions),
            "overall_solo_levels": ["Relational"] if not question.subquestions else ["Unistructural", "Relational"],
            "question_level_summary": "Automatic SOLO analysis failed, so a fallback summary was used.",
            "subquestion_solo_map": {
                str(sq.get("id")): "Relational" for sq in (question.subquestions or [])
            },
        }

    pred.setdefault("question_targets_multiple_levels", False)
    pred.setdefault("overall_solo_levels", [])
    pred.setdefault("question_level_summary", "")
    pred.setdefault("subquestion_solo_map", {})

    return pred, debug_prompt

# Functions for reading CSVs and building the DataFrame for student report input
def read_csv_with_student_identity(csv_text: str) -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(io.StringIO(csv_text)).reset_index(drop=True)
    df["_row_number"] = range(len(df))

    has_real_student_id = False
    if "STUDENT_ID" in df.columns:
        df["STUDENT_ID"] = df["STUDENT_ID"].fillna("").astype(str).str.strip()
        has_real_student_id = df["STUDENT_ID"].ne("").any()
    else:
        df["STUDENT_ID"] = ""

    return df, has_real_student_id


def build_student_report_input_df(criterion_csv_text: str, norm_csv_text: str) -> pd.DataFrame:
    crit_df, crit_has_id = read_csv_with_student_identity(criterion_csv_text)
    norm_df, norm_has_id = read_csv_with_student_identity(norm_csv_text)

    useful_norm_cols = [c for c in [
        "STUDENT_ID",
        "_row_number",
        "overall_class_position",
        "response_length_comparison",
        "sentence_style_comparison",
        "vocabulary_variety_comparison",
        "conceptual_understanding_comparison",
        "precision_of_expression_comparison",
        "conciseness_comparison",
        "strengths_relative_to_class",
        "areas_for_improvement_relative_to_class",
        "norm_referenced_summary",
    ] if c in norm_df.columns]

    norm_small = norm_df[useful_norm_cols].copy()

    if crit_has_id and norm_has_id:
        merged = crit_df.merge(
            norm_small.drop(columns=["_row_number"], errors="ignore"),
            on="STUDENT_ID",
            how="left",
            suffixes=("", "_norm"),
        )
    else:
        merged = crit_df.merge(
            norm_small.drop(columns=["STUDENT_ID"], errors="ignore"),
            on="_row_number",
            how="left",
            suffixes=("", "_norm"),
        )

    merged["display_name"] = [
        sid if str(sid).strip() else f"Student {i+1}"
        for i, sid in enumerate(merged["STUDENT_ID"].tolist
        ())
    ]

    return merged

#Evidence building function for student report
def build_student_report_evidence(row: pd.Series, question: Question, solo_analysis: Dict[str, Any]) -> Dict[str, Any]:
    response_cols = [c for c in row.index if str(c).startswith("STUDENT RESPONSES")]
    response_text = row[response_cols[0]] if response_cols else ""

    sub_scores = {}
    for c in row.index:
        if str(c).startswith("sub_"):
            sid = str(c).replace("sub_", "")
            val = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            sub_scores[sid] = None if pd.isna(val) else float(val)

    sub_feedback = {}
    for c in row.index:
        if str(c).startswith("feedback_") and c != "feedback_overall":
            sid = str(c).replace("feedback_", "")
            sub_feedback[sid] = str(row.get(c, "")).strip()

    return {
        "student_id": row.get("display_name", ""),
        "response_excerpt": truncate_text(response_text, 300),
        "criterion_referenced": {
            "total_score": safe_float(row.get("total_score")),
            "question_max_score": question.max_score,
            "rationale": truncate_text(row.get("rationale", ""), 280),
            "feedback_overall": truncate_text(row.get("feedback_overall", ""), 220),
            "sub_scores": sub_scores,
            "sub_feedback": sub_feedback,
        },
        "norm_referenced": {
            "overall_class_position": row.get("overall_class_position", ""),
            "response_length_comparison": row.get("response_length_comparison", ""),
            "sentence_style_comparison": row.get("sentence_style_comparison", ""),
            "vocabulary_variety_comparison": row.get("vocabulary_variety_comparison", ""),
            "conceptual_understanding_comparison": row.get("conceptual_understanding_comparison", ""),
            "precision_of_expression_comparison": row.get("precision_of_expression_comparison", ""),
            "conciseness_comparison": row.get("conciseness_comparison", ""),
            "strengths_relative_to_class": row.get("strengths_relative_to_class", ""),
            "areas_for_improvement_relative_to_class": row.get("areas_for_improvement_relative_to_class", ""),
            "norm_referenced_summary": truncate_text(row.get("norm_referenced_summary", ""), 220),
        },
        "solo_question_analysis": solo_analysis,
    }

#

def build_student_report_messages(
    question: Question,
    student_evidence: Dict[str, Any],
    discipline: str,
    max_total_bullets_per_student: int = 3,
):
    rubric_summary = build_rubric_summary(question)

    user_text = (
        f"{rubric_summary}\n\n"
        "Write a concise student-friendly report for ONE student.\n"
        "Use the question, rubric, model answer, SOLO analysis, rubrics or model answer-referenced evidence, and norm-referenced evidence.\n\n"

        "IMPORTANT RULES:\n"
        "- Write for a student, not a teacher.\n"
        "- Use plain English.\n"
        "- Speak directly to the student using 'you'.\n"
        "- Be supportive, clear, and brief.\n"
        "- Do NOT use technical assessment terms in the final report.\n"
        "- Do NOT use terms such as 'rubrics-referenced', 'model-answer-referenced', 'norm-referenced', "
        "'SOLO taxonomy', 'prestructural', 'unistructural', 'multistructural', "
        "'relational', or 'extended abstract' in the final report.\n"
        "- If the SOLO analysis suggests different levels of understanding, translate them into simple language.\n"
        "- For example:\n"
        "  * 'identifying one key point'\n"
        "  * 'including several relevant points'\n"
        "  * 'linking ideas together clearly'\n"
        "  * 'applying the idea more broadly'\n"
        "- If the question mainly tests one level of understanding, summarise how the student performed at that level.\n"
        "- If the question tests multiple levels of understanding, explain in simple language which parts the student did better on and which parts were weaker.\n"
        "- Keep the report easy to read quickly.\n"
        "- Keep each bullet short, concrete, and easy to understand.\n"
        "- Avoid vague advice such as 'improve analysis' unless you explain what was missing.\n"
        "- Focus on what the student did well, what was missing, and what would improve the answer.\n"
        "- Do not invent evidence.\n\n"

        "OUTPUT RULES:\n"
        "- The report must include only:\n"
        "  1. overall_summary\n"
        "  2. strong_areas\n"
        "  3. weak_areas\n"
        "- strong_areas and weak_areas must be bullet-point style lists.\n"
        f"- Across strong_areas and weak_areas COMBINED, do not exceed {max_total_bullets_per_student} bullets in total.\n"
        "- Put rubric-based feedback first.\n"
        "- Put class-comparison feedback after rubric-based feedback.\n"
        "- Do not make the report long.\n\n"

        f"Student evidence:\n{json.dumps(student_evidence, indent=2)}\n\n"

        "Return a single valid JSON object with exactly this structure:\n"
        "{\n"
        '  "student_id": string,\n'
        '  "display_name": string,\n'
        '  "overall_summary": string,\n'
        '  "strong_areas": [string, ...],\n'
        '  "weak_areas": [string, ...]\n'
        "}\n"
        "Do not include any text before or after the JSON."
    )

    system_prompt = get_student_report_system_prompt(discipline)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    debug_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_text}"
    return messages, debug_prompt

# Function to enforce total bullet limit across strong and weak areas
def enforce_total_bullet_limit(
    strong_areas: List[str],
    weak_areas: List[str],
    max_total: int = 3,
) -> tuple[List[str], List[str]]:
    strong = [str(x).strip() for x in strong_areas if str(x).strip()]
    weak = [str(x).strip() for x in weak_areas if str(x).strip()]

    ordered = []

    # keep one from each section first if available
    if strong:
        ordered.append(("strong", strong[0]))
    if weak:
        ordered.append(("weak", weak[0]))

    for item in strong[1:]:
        ordered.append(("strong", item))
    for item in weak[1:]:
        ordered.append(("weak", item))

    ordered = ordered[:max_total]

    final_strong = [txt for kind, txt in ordered if kind == "strong"]
    final_weak = [txt for kind, txt in ordered if kind == "weak"]

    return final_strong, final_weak

# Handwriting transcription messages
def build_handwriting_transcription_messages(
    img: Image.Image,
    discipline: str,
) -> tuple[List[Dict[str, Any]], str]:
    system_prompt = (
        f"You are an expert academic handwriting transcription assistant for {discipline}. "
        "Your task is to read a student's handwritten short-answer response and transcribe it into typed text.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Return ONLY a single valid JSON object and nothing else.\n"
        "- Do not add explanation outside the JSON.\n"
        "- Preserve the student's wording as closely as possible.\n"
        "- Preserve numbering such as 1, 2, 3 when visible.\n"
        "- Preserve line breaks where they help readability.\n"
        "- Do NOT improve, rewrite, or correct the answer.\n"
        "- If a word is unclear, use [unclear].\n"
        "- If the page is blank, return an empty transcription.\n"
    )

    user_text = (
        "Transcribe the handwritten student response in this image.\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  "transcription": string,\n'
        '  "confidence": number\n'
        "}\n"
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    debug_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_text}"
    return messages, debug_prompt


API_KEY = os.getenv("BACKEND_API_KEY", "")

def check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

#FastAPI backend
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

#Decode images

@app.post("/grade", response_model=GradeResult)

def grade(req: GradeRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)


#def grade(req: GradeRequest):
    exam_images: List[Image.Image] = []
    for b64 in req.images:
        try:
            img_data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            exam_images.append(img)
        except Exception:
            continue
    user_text = build_user_text(
        question=req.question,
        few_shot_list=req.few_shot,
        target_resp=req.student_response,
        has_subquestions=req.has_subquestions,
        challenge_mode=req.challenge_mode,
        challenge_reason=req.challenge_reason,
        original_total_score=req.original_total_score,
        original_sub_scores=req.original_sub_scores,)
    
    system_prompt=get_system_prompt(req.discipline)
    
    debug_prompt = (
        "SYSTEM PROMPT:\n"
        f"{system_prompt}\n\n"
        "USER PROMPT:\n"
        f"{user_text}")

    messages = build_messages(
    question=req.question,
    exam_images=exam_images,
    few_shot_list=req.few_shot,
    target_resp=req.student_response,
    has_subquestions=req.has_subquestions,
    discipline=req.discipline,
    challenge_mode=req.challenge_mode,
    challenge_reason=req.challenge_reason,
    original_total_score=req.original_total_score,
    original_sub_scores=req.original_sub_scores,
)


    raw_output = call_qwen_vl(messages, max_new_tokens=1000)

    try:
        pred = extract_json(raw_output)
    except Exception as e:
        return GradeResult(
            response_id=req.student_response.response_id,
            total_score=0.0,
            sub_scores={},
            rationale=f"Failed to parse model output: {e}",
            feedback={"_overall": "Vox-LM grading failed; please grade manually."},
            confidence=0.0,
            highlights={},
            debug_prompt=debug_prompt,
            challenge_review=(
                "Challenge reviewed, but re-marking by Vox-LM failed."
                if req.challenge_mode else ""
            ),
            challenged=req.challenge_mode,
            original_total_score=req.original_total_score if req.challenge_mode else None,
            missing_key_point="",
            needs_review=True,
            review_reasons=["Model output parsing failure. Manual review needed."],
        )


    pred["response_id"] = req.student_response.response_id

# Defaults
    pred.setdefault("total_score", 0.0)
    pred.setdefault("sub_scores", {})
    pred.setdefault("rationale", "")
    pred.setdefault("feedback", {})
    pred.setdefault("confidence", 50.0)
    pred.setdefault("highlights", {})
    # Normalise highlight categories for backward compatibility.
    if isinstance(pred.get("highlights"), dict):
        for sid, h in pred["highlights"].items():
            if not isinstance(h, dict):
                pred["highlights"][sid] = {
                    "correct": [],
                    "out_of_scope": [],
                    "misconception": [],
                    "uncertain": [],
                }
                continue

            h.setdefault("correct", [])
            h.setdefault("out_of_scope", [])
            h.setdefault("misconception", [])
            h.setdefault("uncertain", [])

            # Old outputs may still use "incorrect".
            if "incorrect" in h and not h.get("misconception"):
                h["misconception"] = h.get("incorrect", [])

            # Remove old key if you want cleaner output.
            h.pop("incorrect", None)

    pred.setdefault("missing_key_point", "")
    if not req.challenge_mode:
        pred["challenge_review"] = ""
    
    pred.setdefault("marking_breakdown", [])

    # If there are no subquestions but the model returned an itemised marking breakdown,
    # use the sum of awarded item marks as the authoritative total.
    if not req.has_subquestions and isinstance(pred.get("marking_breakdown"), list):
        try:
            breakdown_sum = 0.0
            valid_items = 0

            for item in pred["marking_breakdown"]:
                if not isinstance(item, dict):
                    continue
                awarded = item.get("awarded", None)
                if awarded is None:
                    continue
                breakdown_sum += float(awarded)
                valid_items += 1

            if valid_items > 0:
                pred["total_score"] = breakdown_sum
        except Exception:
            pass



    if not req.has_subquestions:
        pred["sub_scores"] = {}

    try:
        conf = float(pred["confidence"])
    except Exception:
        conf = 50.0  # default if parsing fails
    conf = max(0.0, min(100.0, conf))
    pred["confidence"] = conf

#I will include this in main code to enforce consistency between total_score and sub_scores
    if req.has_subquestions and isinstance(pred.get("sub_scores"), dict):
        try:
            pred["total_score"] = sum(float(v) for v in pred["sub_scores"].values())
        except Exception:
            pass

    # Clamp total score to valid range
    try:
        total = float(pred.get("total_score", 0.0))
        if req.question.max_score is not None:
            qmax = float(req.question.max_score)
            total = max(0.0, min(qmax, total))
        else:
            total = max(0.0, total)
        pred["total_score"] = total
    except Exception:
        pred["total_score"] = 0.0

    pred["challenged"] = bool(req.challenge_mode)
    pred["original_total_score"] = req.original_total_score if req.challenge_mode else None

    if req.challenge_mode and req.original_total_score is not None:
        try:
            original_total = float(req.original_total_score)
            new_total = float(pred.get("total_score", 0.0))

            # Challenge is not allowed to reduce the original mark
            if new_total <= original_total:
                pred["total_score"] = original_total

                if isinstance(req.original_sub_scores, dict):
                    pred["sub_scores"] = req.original_sub_scores

                if not str(pred.get("challenge_review", "")).strip():
                    pred["challenge_review"] = (
                        "Challenge reviewed. No extra credit was awarded because the reasons given "
                        "were not sufficiently supported by the original submitted answer."
                    )
            else:
                if not str(pred.get("challenge_review", "")).strip():
                    pred["challenge_review"] = (
                        "Challenge reviewed. Extra credit was awarded because the reasons pointed to "
                        "material that was already present in the original submitted answer."
                    )
        except Exception:
            pass
    #clamp after challenge handling
    try:
        total = float(pred.get("total_score", 0.0))
        if req.question.max_score is not None:
            qmax = float(req.question.max_score)
            total = max(0.0, min(qmax, total))
        else:
            total = max(0.0, total)
        pred["total_score"] = total
    except Exception:
        pred["total_score"] = 0.0
    
    sync_rationale_with_final_score(req, pred)

    needs_review, review_reasons = compute_needs_review(req, pred)
    pred["needs_review"] = needs_review
    pred["review_reasons"] = review_reasons

    pred["debug_prompt"] = debug_prompt
    return GradeResult(**pred)

#Path for norm-referenced batch analysis
@app.post("/norm_reference_batch", response_model=BatchNormReferenceResult)
def norm_reference_batch(req: BatchNormReferenceRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
#def norm_reference_batch(req: BatchNormReferenceRequest):
    try:
        teacher_df, diagnostic_df = build_norm_reference_dfs(req.csv_text, req.question)

        teacher_rows = teacher_df.where(pd.notnull(teacher_df), None).to_dict(orient="records")
        diagnostic_rows = diagnostic_df.where(pd.notnull(diagnostic_df), None).to_dict(orient="records")

        return BatchNormReferenceResult(
            teacher_rows=teacher_rows,
            diagnostic_rows=diagnostic_rows,
        )
    
    except Exception as e:
        logger.exception("Norm reference batch generation failed")
        raise HTTPException(status_code=500, detail=f"Norm-referenced analysis failed: {e}")



#Decode batch summary data and build messages
@app.post("/summarize_batch", response_model=BatchSummaryResult)
def summarize_batch(req: BatchSummaryRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)



#def summarize_batch(req: BatchSummaryRequest):
    try:
        messages, debug_prompt, local_data = build_batch_summary_messages(
            csv_text=req.csv_text,
            question=req.question,
            has_subquestions=req.has_subquestions,
            discipline=req.discipline,
            max_examples_per_tier=req.max_examples_per_tier,
        )
    except Exception as e:
        logger.exception("Failed to build local summary data")
        return BatchSummaryResult(
            total_students=0,
            scored_students=0,
            class_average=0.0,
            median_score=0.0,
            max_score=0.0,
            min_score=0.0,
            std_score=0.0,
            tier_thresholds={"high": ">= 70%", "mid": "50% to < 70%", "low": "< 50%"},
            tier_counts={"high": 0, "mid": 0, "low": 0},
            score_distribution={"0-24%": 0, "25-49%": 0, "50-74%": 0, "75-100%": 0},
            overall_subquestion_stats={},
            tier_summaries={"high": "", "mid": "", "low": ""},
            strengths=[],
            common_misconceptions=[],
            out_of_scope_points=[],
            weak_areas=[],
            teacher_next_steps=["The automatic summary failed. Please review the graded CSV manually."],
            narrative_summary=f"Failed to compute summary data: {e}",
            subquestion_diagnostics={},
            debug_prompt="",
        )

    try:
        raw_output = call_qwen_vl(messages, max_new_tokens=2000)
        pred = extract_json(raw_output)
    except Exception as e:
        logger.exception("Model summary generation failed")
        pred = {
            "tier_summaries": {"high": "", "mid": "", "low": ""},
            "strengths": [],
            "common_misconceptions": [],
            "out_of_scope_points": [],
            "weak_areas": [],
            "teacher_next_steps": [
                "Automatic qualitative summarization failed. Use the exact local class statistics and review representative scripts manually."
            ],
            "narrative_summary": f"Local statistics were computed successfully, but model-based interpretation failed: {e}",
        }

    pred.setdefault("tier_summaries", {"high": "", "mid": "", "low": ""})
    pred.setdefault("strengths", [])
    pred.setdefault("common_misconceptions", [])
    pred.setdefault("out_of_scope_points", [])
    pred.setdefault("weak_areas", [])
    pred.setdefault("teacher_next_steps", [])
    pred.setdefault("narrative_summary", "")
    pred.setdefault("subquestion_diagnostics", {})

    pred["total_students"] = local_data["total_students"]
    pred["scored_students"] = local_data["scored_students"]
    pred["class_average"] = local_data["class_average"]
    pred["median_score"] = local_data["median_score"]
    pred["max_score"] = local_data["max_score"]
    pred["min_score"] = local_data["min_score"]
    pred["std_score"] = local_data["std_score"]
    pred["tier_thresholds"] = local_data["tier_thresholds"]
    pred["tier_counts"] = local_data["tier_counts"]
    pred["score_distribution"] = local_data["score_distribution"]
    pred["overall_subquestion_stats"] = local_data["overall_subquestion_stats"]
    pred["subquestion_diagnostics"] = pred.get("subquestion_diagnostics", {}) or {}

    if not pred.get("common_misconceptions"):
        pred["common_misconceptions"] = local_data.get("observed_misconception_patterns", [])

    if not pred.get("out_of_scope_points"):
        pred["out_of_scope_points"] = local_data.get("observed_out_of_scope_patterns", [])
    
    pred["debug_prompt"] = debug_prompt

    return BatchSummaryResult(**pred)


# Path for student report batch generation
@app.post("/student_reports_batch", response_model=StudentReportBatchResult)
def student_reports_batch(req: StudentReportRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
#def student_reports_batch(req: StudentReportRequest):
    try:
        solo_analysis, solo_debug_prompt = analyse_question_solo(req.question, req.discipline)
        merged_df = build_student_report_input_df(req.criterion_csv_text, req.norm_csv_text)
    except Exception as e:
        logger.exception("Failed to prepare student report inputs")
        raise HTTPException(status_code=500, detail=f"Failed to prepare student reports: {e}")

    reports: List[StudentReportItem] = []
    debug_prompt_sample = ""

    for idx, row in merged_df.iterrows():
        student_id = str(row.get("display_name", f"student_{idx+1}")).strip() or f"student_{idx+1}"

        student_evidence = build_student_report_evidence(row, req.question, solo_analysis)
        messages, debug_prompt = build_student_report_messages(
            question=req.question,
            student_evidence=student_evidence,
            discipline=req.discipline,
            max_total_bullets_per_student=req.max_total_bullets_per_student,
        )

        if not debug_prompt_sample:
            debug_prompt_sample = (
                "SOLO ANALYSIS PROMPT:\n"
                f"{solo_debug_prompt}\n\n"
                "FIRST STUDENT REPORT PROMPT:\n"
                f"{debug_prompt}"
            )

        try:
            raw_output = call_qwen_vl(messages, max_new_tokens=800)
            pred = extract_json(raw_output)
        except Exception as e:
            pred = {
                "student_id": student_id,
                "display_name": student_id,
                "overall_summary": f"A personalised report could not be generated automatically: {e}",
                "strong_areas": [],
                "weak_areas": [],}

        pred["student_id"] = student_id
        pred["display_name"] = student_id
        pred.setdefault("overall_summary", "")
        pred.setdefault("strong_areas", [])
        pred.setdefault("weak_areas", [])

        pred["strong_areas"], pred["weak_areas"] = enforce_total_bullet_limit(
            pred.get("strong_areas", []),
            pred.get("weak_areas", []),
            max_total=req.max_total_bullets_per_student,)

        pred["overall_summary"] = simplify_student_report_text(pred.get("overall_summary", ""))

        pred["strong_areas"] = [
            simplify_student_report_text(x) for x in pred.get("strong_areas", [])]

        pred["weak_areas"] = [
            simplify_student_report_text(x) for x in pred.get("weak_areas", [])]

        pred["report_text"] = (
            f"{pred['display_name']}\n\n"
            f"Summary: {pred['overall_summary']}\n\n"
            f"Strong areas:\n"
            + ("\n".join(f"- {x}" for x in pred["strong_areas"]) if pred["strong_areas"] else "- None identified")
            + "\n\nWeak areas:\n"
            + ("\n".join(f"- {x}" for x in pred["weak_areas"]) if pred["weak_areas"] else "- None identified")
        )

        reports.append(StudentReportItem(**pred))

    return StudentReportBatchResult(
        solo_question_analysis=solo_analysis,
        reports=reports,
        debug_prompt_sample=debug_prompt_sample,
    )

# Path for handwriting transcription
@app.post("/transcribe_handwriting", response_model=HandwritingTranscriptionResult)
def transcribe_handwriting(req: HandwritingTranscriptionRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
#def transcribe_handwriting(req: HandwritingTranscriptionRequest):
    try:
        img_data = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    messages, debug_prompt = build_handwriting_transcription_messages(
        img=img,
        discipline=req.discipline,
    )

    try:
        raw_output = call_qwen_vl(messages, max_new_tokens=1200)
        pred = extract_json(raw_output)
    except Exception as e:
        logger.exception("Handwriting transcription failed")
        return HandwritingTranscriptionResult(
            transcription="",
            confidence=0.0,
            debug_prompt=debug_prompt,
        )

    pred.setdefault("transcription", "")
    pred.setdefault("confidence", 50.0)

    try:
        conf = float(pred["confidence"])
    except Exception:
        conf = 50.0

    conf = max(0.0, min(100.0, conf))

    return HandwritingTranscriptionResult(
        transcription=str(pred.get("transcription", "") or ""),
        confidence=conf,
        debug_prompt=debug_prompt,
    )

# Path for Vox-LM chat interactions
@app.post("/voxlm_chat", response_model=VoxChatResult)
def voxlm_chat(req: VoxChatRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)

    messages, debug_prompt = build_voxlm_chat_messages(req)

    try:
        raw_output = call_qwen_vl(messages, max_new_tokens=700)
        assistant_message = str(raw_output or "").strip()
    except Exception as e:
        logger.exception("Vox-LM chat failed")
        assistant_message = (
            "I am Vox-LM. Sorry, I could not generate a response just now. "
            "Please ask your teacher to review the feedback with you."
        )

    return VoxChatResult(
        assistant_message=assistant_message,
        debug_prompt=debug_prompt,
    )

# Path for model answer refinement
@app.post("/refine_model_answer", response_model=RefineModelAnswerResult)
def refine_model_answer(req: RefineModelAnswerRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)

    messages, debug_prompt = build_refine_model_answer_messages(req)

    try:
        raw_output = call_qwen_vl(messages, max_new_tokens=1400)
        pred = extract_json(raw_output)
    except Exception as e:
        logger.exception("Model answer refinement failed")
        pred = {
            "rating_score": 0.0,
            "rating_label": "Unable to assess",
            "strengths": [],
            "issues": [f"Automatic model answer review failed: {e}"],
            "rewritten_model_answer": "",
            "suggested_marking_structure": "",
        }

    pred.setdefault("rating_score", 0.0)
    pred.setdefault("rating_label", "")
    pred.setdefault("strengths", [])
    pred.setdefault("issues", [])
    pred.setdefault("rewritten_model_answer", "")
    pred.setdefault("suggested_marking_structure", "")

    try:
        pred["rating_score"] = max(0.0, min(100.0, float(pred["rating_score"])))
    except Exception:
        pred["rating_score"] = 0.0

    pred["debug_prompt"] = debug_prompt

    return RefineModelAnswerResult(**pred)

# Path for MCQ generation from video lectures
@app.post("/generate/mcq_from_videos", response_model=VideoMCQGenerateResult)
def generate_video_mcq(
    video: UploadFile = File(...),
    discipline: Literal["dentistry", "medicine", "law", "education"] = Form("dentistry"),
    topic: str = Form(""),
    target_level: str = Form(""),
    learning_objectives_json: str = Form("[]"),
    num_check_questions: int = Form(4),
    allow_anticipatory: bool = Form(True),
    use_frame_analysis: bool = Form(False),
    frame_interval_seconds: int = Form(30),
    max_frames: int = Form(20),
    chunk_seconds: int = Form(30),
    x_api_key: Optional[str] = Header(default=None),
):

    check_api_key(x_api_key)

    warnings: List[str] = []
    video_id = f"vid_{uuid.uuid4().hex[:12]}"

    video_path = ""
    audio_path = ""

    try:
        try:
            learning_objectives = json.loads(learning_objectives_json)
            if not isinstance(learning_objectives, list):
                learning_objectives = []
            learning_objectives = [str(x).strip() for x in learning_objectives if str(x).strip()]
        except Exception:
            learning_objectives = []
            warnings.append("Could not parse learning objectives JSON; using an empty list.")

        nnum_check_questions = max(1, min(10, int(num_check_questions)))
        frame_interval_seconds = max(5, min(120, int(frame_interval_seconds)))
        max_frames = max(1, min(60, int(max_frames)))
        chunk_seconds = max(15, min(120, int(chunk_seconds)))

        logger.info("Saving uploaded video...")
        video_path = save_upload_to_tempfile(video)

        duration_seconds = get_video_duration_seconds(video_path)
        if duration_seconds <= 0:
            warnings.append("Could not determine video duration.")

        logger.info("Extracting audio from video...")
        audio_path = extract_audio_to_wav(video_path)

        logger.info("Transcribing audio...")
        transcript_segments = transcribe_audio_with_timestamps(audio_path)

        if not transcript_segments:
            warnings.append("No transcript segments were produced. MCQ generation may be weak.")

        frame_summaries: List[Dict[str, Any]] = []

        if use_frame_analysis:
            logger.info("Extracting video frames for visual analysis...")
            frames = extract_video_frames(
                video_path=video_path,
                interval_seconds=frame_interval_seconds,
                max_frames=max_frames,
            )

            if not frames:
                warnings.append("No frames were extracted for visual analysis.")
            else:
                logger.info("Analysing %d video frames...", len(frames))
                frame_summaries = summarise_video_frames(
                    frames=frames,
                    discipline=discipline,
                )
        else:
            warnings.append("Frame analysis was disabled.")

        enriched_segments = build_enriched_video_evidence(
            transcript_segments=transcript_segments,
            frame_summaries=frame_summaries,
            chunk_seconds=chunk_seconds)

        candidate_anticipatory_moments = extract_candidate_anticipatory_moments(
            transcript_segments=transcript_segments,
            max_candidates=8, min_question_time_seconds=30.0)

        if allow_anticipatory and not candidate_anticipatory_moments:
            warnings.append(
                "Anticipatory questions were allowed, but no strong teacher question moments were automatically detected.")


        messages, debug_prompt = build_video_mcq_generation_messages(
            discipline=discipline,
            topic=topic,
            target_level=target_level,
            learning_objectives=learning_objectives,
            duration_seconds=duration_seconds,
            enriched_segments=enriched_segments,
            num_check_questions=num_check_questions,
            allow_anticipatory=allow_anticipatory,
            candidate_anticipatory_moments=candidate_anticipatory_moments)


        logger.info("Generating video MCQs...")
        raw_output = call_video_llm(messages, max_new_tokens=20000)

        try:
            pred = extract_json(raw_output)
        except Exception as e:
            logger.exception("Video MCQ JSON parsing failed")
            raise HTTPException(
                status_code=500,
                detail=f"Video MCQ generation failed because model output was not valid JSON: {e}",
            )

        pred = normalise_video_mcq_prediction(
            pred=pred,
            num_check_questions=num_check_questions)

        if allow_anticipatory:
            anticipatory_count = count_anticipatory_questions(pred)

            if candidate_anticipatory_moments and anticipatory_count == 0:
                warnings.append(
                    "First MCQ generation pass produced no anticipatory questions despite candidate anticipatory moments. Retrying once."
                )

                retry_instruction = (
                    "\n\nCRITICAL RETRY INSTRUCTION:\n"
                    "Your previous response generated no anticipatory questions even though candidate anticipatory moments were provided.\n"
                    "Regenerate the JSON.\n"
                    "At least one item in embedded_questions MUST have question_type set to \"anticipatory\".\n"
                    "Use one of the provided candidate anticipatory moments.\n"
                    "The anticipatory question timestamp_seconds must match the candidate question_timestamp_seconds.\n"
                    "The question should ask the learner to predict what the teacher is about to explain or reveal.\n"
                    "Do not label all in-video questions as embedded_check.\n"
                    "Return only the JSON object.\n"
                )

                try:
                    retry_messages = copy.deepcopy(messages)

                    for content_item in retry_messages[-1]["content"]:
                        if content_item.get("type") == "text":
                            content_item["text"] += retry_instruction
                            break

                    raw_output_retry = call_video_llm(
                        retry_messages,
                        max_new_tokens=20000,
                    )

                    pred_retry = extract_json(raw_output_retry)

                    pred_retry = normalise_video_mcq_prediction(
                        pred=pred_retry,
                        num_check_questions=num_check_questions,
                    )

                    if count_anticipatory_questions(pred_retry) > 0:
                        pred = pred_retry
                        debug_prompt += "\n\nRETRY INSTRUCTION USED:\n" + retry_instruction
                    else:
                        warnings.append(
                            "Retry still produced no anticipatory questions. Keeping first-pass output."
                        )

                except Exception as e:
                    warnings.append(f"Retry for anticipatory questions failed: {e}")


        if allow_anticipatory:
            anticipatory_count = count_anticipatory_questions(pred)

            if candidate_anticipatory_moments and anticipatory_count == 0:
                warnings.append(
                    "Anticipatory questions were allowed and candidate teacher question moments were detected, "
                    "but the final output still generated only embedded questions. Teacher review recommended!")


        return VideoMCQGenerateResult(
            video_id=video_id,
            filename=video.filename or "uploaded_video",
            duration_seconds=round(float(duration_seconds), 2),
            transcript_summary=pred.get("transcript_summary", ""),
            transcript_segments=transcript_segments,
            frame_summaries=frame_summaries,
            teaching_segments=pred.get("teaching_segments", []),
            pre_question=pred.get("pre_question", {}),
            embedded_questions=pred.get("embedded_questions", []),
            warnings=warnings,
            debug_prompt=debug_prompt,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Video MCQ generation failed")
        raise HTTPException(status_code=500, detail=f"Video MCQ generation failed: {e}")

    finally:
        for path in [video_path, audio_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


