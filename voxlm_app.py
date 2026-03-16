import base64
import json
import re
from typing import Dict, List
import requests
import streamlit as st
import io
import pandas as pd

#st.set_option("global.dataFrameSerialization", "legacy")

#FastAPI connect
BACKEND_URL = "http://localhost:8000/grade"

#df sanitizer
def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame(df.to_dict(orient="records"))
    clean = clean.astype(object)
    clean = clean.where(pd.notnull(clean), None)
    clean.columns = [str(c) for c in clean.columns]
    return clean

#Three panel Vox-LM Prototype

#Highlighter for mid panel
def build_highlighted_html(
    text: str,
    correct_spans: List[str],
    incorrect_spans: List[str],
    uncertain_spans: List[str],
) -> str:
    """
    Simple span-based highlighter.
    Correct phrases -> green background
    Incorrect phrases -> red background
    Uncertain phrases -> yellow background
    """
    import html

    safe_text = html.escape(text)

    def unique_sorted(spans: List[str]) -> List[str]:
        spans = list(set(s for s in spans if s and s.strip()))
        spans.sort(key=len, reverse=True)
        return spans

    replacements = []

    for phrase in unique_sorted(incorrect_spans):
        esc = html.escape(phrase)
        replacements.append(
            (esc, f'<span style="background-color:#f7c5c5">{esc}</span>')  # red hex style
        )

    for phrase in unique_sorted(correct_spans):
        esc = html.escape(phrase)
        replacements.append(
            (esc, f'<span style="background-color:#c8f7c5">{esc}</span>')  # green hex style
        )

    for phrase in unique_sorted(uncertain_spans):
        esc = html.escape(phrase)
        replacements.append(
            (esc, f'<span style="background-color:#fff5c5">{esc}</span>')  # yellow hex style
        )

    for src, dst in replacements:
        safe_text = safe_text.replace(src, dst)

    return safe_text


#parsers for text box
def parse_subquestions_from_text(text: str) -> List[Dict]:
    """
    Parse subquestions from free text.

    Expected structure per subquestion block (separated by blank lines):

    1:
    Question text
    Sub-score for question
    Rubric / model answer (optional, can span multiple lines)

    First line may be of the form: "1", "1:", "1.", or "1)".
    The ID is normalised to the digits only (e.g. "1").
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    subqs: List[Dict] = []

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        first = lines[0]
        # Normalise ID: accept "1", "1:", "1.", "1)"
        m = re.match(r"^(\d+)\s*[:\.\)\-]?\s*$", first)
        if m:
            sid = m.group(1)  # just digits, e.g. "1"
        else:
            sid = first.strip()

        prompt = lines[1]

        try:
            max_score = float(lines[2])
        except Exception:
            max_score = None

        rubric = ""
        if len(lines) > 3:
            rubric = "\n".join(lines[3:])

        subqs.append(
            {
                "id": sid,
                "prompt": prompt,
                "max_score": max_score,
                "rubric": rubric,
            }
        )

    return subqs


def parse_few_shot_from_text(
    text: str,
    has_subquestions: bool,
    subquestions: List[Dict],
) -> List[Dict]:
    """
    Parse few-shot examples from free text.

    Expected structure per example block (separated by blank lines):

    Student Number: S1
    Answer:
    <student's full answer, possibly multiple lines>
    Marker score: 4.5

    If subquestions exist, the same answer text is used
    for every subquestion id.
    If no subquestions, stored under key "overall".
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    few_shot: List[Dict] = []

    for block in blocks:
        lines = block.splitlines()
        student_id = None
        answer_lines: List[str] = []
        score = None

        idx_answer_start = None
        idx_marker = None

        for i, line in enumerate(lines):
            l = line.strip()
            lower = l.lower()
            if lower.startswith("student number:"):
                student_id = l.split(":", 1)[1].strip() or f"student_{len(few_shot) + 1}"
            elif lower.startswith("answer:"):
                idx_answer_start = i + 1
            elif lower.startswith("marker score:"):
                idx_marker = i
                try:
                    score_str = l.split(":", 1)[1].strip()
                    score = float(score_str)
                except Exception:
                    score = 0.0

        if idx_answer_start is not None:
            if idx_marker is None:
                answer_lines = [l.strip() for l in lines[idx_answer_start:]]
            else:
                answer_lines = [l.strip() for l in lines[idx_answer_start:idx_marker]]

        answer_text = "\n".join(answer_lines).strip()

        if not student_id and not answer_text:
            continue

        if score is None:
            score = 0.0

        if has_subquestions and subquestions:
            answers = {str(sq.get("id")): answer_text for sq in subquestions}
        else:
            answers = {"overall": answer_text}

        few_shot.append(
            {
                "response_id": student_id or f"student_{len(few_shot) + 1}",
                "answers": answers,
                "marker_score": score,
            }
        )

    return few_shot


def parse_student_answers_from_text(
    text: str,
    subquestions: List[Dict],
) -> Dict[str, str]:
    """
    Parse student answers into a dict keyed by subquestion id.

    Uses the IDs defined in `subquestions` and supports answer formats like:

    1:
    The length is around 9mm

    2. I will use a molar size pontic for 36 ...

    3) As the pontic will be molar sized ...

    4 The path of insertion will be tilted mesially and palatally.

    Blank lines are allowed but not required. If a subquestion
    receives no text, its value will be an empty string.
    """
    valid_ids = {str(sq.get("id")) for sq in subquestions}

    # Accumulate lines per subquestion
    answers_lines: Dict[str, List[str]] = {sid: [] for sid in valid_ids}
    current_sid: str | None = None

    # Header regex:
    #  "1:"      -> id="1", rest=""
    #  "1. ans"  -> id="1", rest="ans"
    #  "2) ans"  -> id="2", rest="ans"
    #  "3 ans"   -> id="3", rest="ans"
    header_re = re.compile(r"^(\d+)\s*[:\.\)\-]?\s*(.*)$")

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            if current_sid is not None:
                answers_lines[current_sid].append("")
            continue

        m = header_re.match(stripped)
        if m:
            sid_candidate, rest = m.group(1), m.group(2)
            if sid_candidate in valid_ids:
                current_sid = sid_candidate
                if rest:
                    answers_lines[current_sid].append(rest)
                continue

        if current_sid is not None:
            answers_lines[current_sid].append(line)

    answers: Dict[str, str] = {}
    for sq in subquestions:
        sid = str(sq.get("id"))
        joined = "\n".join(answers_lines.get(sid, [])).strip()
        answers[sid] = joined

    return answers


#Streamlit frontend
st.set_page_config(layout="wide", page_title="Vox-LM SAQ Marking Prototype for Vox 2.0")
st.title(":blue[Vox-LM] _(SAQ Prototype)_")

#defaults
if "grade_result" not in st.session_state:
    st.session_state.grade_result = None

if "override_mode" not in st.session_state:
    st.session_state.override_mode = False

if "q_edit_mode" not in st.session_state:
    st.session_state.q_edit_mode = True

if "discipline_choice" not in st.session_state:
    st.session_state.discipline_choice = "Dentistry"

discipline = st.session_state.discipline_choice.lower()

if "batch_results_df" not in st.session_state:
    st.session_state.batch_results_df = None

#sidebar
with st.sidebar:
    st.header("Vox-LM Controls")

    discipline_label= st.radio(
        "Faculty",
        ["Dentistry", "Law", "Education"],
        index=["Dentistry", "Law", "Education"].index(st.session_state.discipline_choice),
        key="discipline_choice",
    )

    if st.button("**:red[Human override]**", key="btn_override"):
        st.session_state.override_mode = not st.session_state.override_mode

    st.write(f"Override mode: {'ON' if st.session_state.override_mode else 'OFF'}")

    if st.button("**:blue[Question / rubric editor]**", key="btn_qedit"):
        st.session_state.q_edit_mode = not st.session_state.q_edit_mode

    st.write(f"Edit question/rubric mode: {'ON' if st.session_state.q_edit_mode else 'OFF'}")
    st.markdown("---")
    st.header("Batch grading")

    batch_csv = st.file_uploader(
        "**:green[Upload CSV of student responses. Title of column in CSV must start with 'STUDENT RESPONSES']**",
        type=["csv"],
        key="batch_csv_uploader"
    )

    batch_grade_btn = st.button("Grade uploaded CSV", key="btn_batch_grade")

discipline = st.session_state.discipline_choice.lower()
override = st.session_state.override_mode
q_edit_mode = st.session_state.q_edit_mode

#three column panel
col_left, col_mid, col_right = st.columns(3)

#left
with col_left:
    st.subheader(":violet[Question, Rubric, Model Answer, and Examples]")

    has_subquestions = st.checkbox("**:blue[Question has subquestions]**", value=False)

    disabled_left = not st.session_state.q_edit_mode

    question_stem = st.text_area(
        ":blue[Question stem]",
        height=150,
        disabled=disabled_left,
    )

    model_answer = st.text_area(
        ":blue[Model answer]",
        height=100,
        disabled=disabled_left,
    )

    global_rubric = st.text_area(
        ":blue[Rubric / marking scheme (optional)]",
        height=100,
        disabled=disabled_left,
    )

    max_score = st.number_input(
        ":blue[Total score]",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        disabled=disabled_left,
    )

    if has_subquestions:
        st.markdown("**:blue[Subquestions format]**")
        st.caption(
            "Format each subquestion as:  \n"
            "Number and colon (e.g., 1:, 2:, 3:, etc.)  \n"
            "Question text  \n"
            "Sub-score for question  \n"
            "Rubric/model answer\n\n"
            "Start separate subquestions with a blank line.\n"
            "First line may also be '1', '1.' or '1)'."
        )

        subq_text = st.text_area(
            ":blue[Subquestions]",
            height=220,
            disabled=disabled_left,
        )
    else:
        subq_text = ""
        st.info("Subquestions disabled; sub-scores will show as NA.")
    
    parsed_subquestions: List[Dict] = []
    if has_subquestions and subq_text.strip():
        try:
            parsed_subquestions = parse_subquestions_from_text(subq_text)
            if not parsed_subquestions:
                st.warning("No valid subquestions parsed. Check the text format.")
        except Exception as e:
            st.error(f"Error parsing subquestions text: {e}")
            parsed_subquestions = []    

    images_files = st.file_uploader(
        ":blue[Upload SAQ images (optional)]",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        disabled=disabled_left,
    )

    st.markdown("**:blue[Few-shot examples (optional)]**")
    st.caption(
        "Each example:  \n"
        "Student Number: [ID]  \n"
        "Answer: [student answers]  \n"
        "Marker score: [number]\n\n"
        "**Separate examples with a blank line.**"
    )

    few_shot_text = st.text_area(
        ":blue[Enter examples here]",
        height=220,
        disabled=disabled_left,
    )


#middle panel
with col_mid:
    st.subheader(":violet[Student Response]")

    # Parse subquestions from text and build the prompt for the target response

    student_answer_text = st.text_area(
        "Answer box",
        height=250,
        key="student_answer",
    )

    response_id = st.text_input("Student name (optional)", value="John Doe")

    if st.button("Grade student"):
        # Build question object
        question_dict = {
            "exam_id": "EXAM",
            "question_id": "Q1",
            "stem": question_stem,
            "max_score": max_score,
            "subquestions": parsed_subquestions if has_subquestions else [],
            "model_answer": model_answer,
            "rubric": global_rubric,
        }

        # Few-shot list from text
        few_shot_list: List[Dict] = []
        if few_shot_text.strip():
            try:
                few_shot_list = parse_few_shot_from_text(
                    few_shot_text,
                    has_subquestions=has_subquestions,
                    subquestions=parsed_subquestions,
                )
            except Exception as e:
                st.error(f"Error parsing few-shot examples: {e}")
                few_shot_list = []

        # Images in base64 form
        images_b64: List[str] = []
        if images_files:
            for f in images_files:
                b = f.read()
                images_b64.append(base64.b64encode(b).decode("utf-8"))

        # Build payload and call backend
        if has_subquestions and parsed_subquestions:
            answers = parse_student_answers_from_text(
                student_answer_text, parsed_subquestions
            )
        else:
            answers = {"overall": student_answer_text}

        payload = {
            "question": question_dict,
            "few_shot": few_shot_list,
            "student_response": {
                "response_id": response_id,
                "answers": answers,
            },
            "has_subquestions": has_subquestions,
            "images": images_b64,
            "discipline": discipline
        }

        try:
            res = requests.post(BACKEND_URL, json=payload, timeout=300)
            if res.status_code != 200:
                st.error(f"Backend error: {res.status_code} {res.text}")
            else:
                st.session_state.grade_result = res.json()
        except Exception as e:
            st.error(f"Request to backend failed: {e}")

    # Highlight student answers
    if st.session_state.grade_result is not None:
        st.markdown("---")
        st.markdown("**Highlighted student answer**")

        result = st.session_state.grade_result
        highlights = result.get("highlights", {})

        all_correct: List[str] = []
        all_incorrect: List[str] = []
        all_uncertain: List[str] = []

        for sid, h in (highlights or {}).items():
            all_correct.extend(h.get("correct", []))
            all_incorrect.extend(h.get("incorrect", []))
            all_uncertain.extend(h.get("uncertain", []))

        html = build_highlighted_html(
            student_answer_text,
            all_correct,
            all_incorrect,
            all_uncertain,
        )
        st.markdown(html, unsafe_allow_html=True)

#bATCH GRADING FRONT

if batch_csv is not None and batch_grade_btn:
    try:
        try:
            df = pd.read_csv(batch_csv, dtype_backend="numpy_nullable")
        except TypeError:
            df = pd.read_csv(batch_csv)

        if "STUDENT RESPONSES" not in df.columns:
            st.error("CSV must contain a 'STUDENT RESPONSES' column.")
        else:
            has_student_id_col = "STUDENT_ID" in df.columns

            question_dict = {
                "exam_id": "EXAM",
                "question_id": "Q1",
                "stem": question_stem,
                "max_score": max_score,
                "subquestions": parsed_subquestions if has_subquestions else [],
                "model_answer": model_answer,
                "rubric": global_rubric,
            }

            few_shot_list: List[Dict] = []
            if few_shot_text.strip():
                try:
                    few_shot_list = parse_few_shot_from_text(
                        few_shot_text,
                        has_subquestions=has_subquestions,
                        subquestions=parsed_subquestions,
                    )
                except Exception as e:
                    st.error(f"Error parsing few-shot examples: {e}")
                    few_shot_list = []

            images_b64: List[str] = []
            if images_files:
                for f in images_files:
                    b = f.read()
                    images_b64.append(base64.b64encode(b).decode("utf-8"))

            results_df = df.copy()
            results_df["total_score"] = None
            results_df["confidence"] = None
            results_df["rationale"] = None
            results_df["feedback_overall"] = None

            # Pre-create sub-score columns if subquestions exist
            if has_subquestions and parsed_subquestions:
                for sq in parsed_subquestions:
                    sid = str(sq.get("id"))
                    col_name = f"sub_{sid}"
                    results_df[col_name] = None

            total_rows = len(df)
            progress = st.progress(0.0)
            status_text = st.empty()

            for i, row in df.iterrows():
                student_text = str(row["STUDENT RESPONSES"] or "")

                # Student ID
                if has_student_id_col:
                    raw_id = row["STUDENT_ID"]
                    student_id = str(raw_id) if pd.notna(raw_id) else f"student_{i+1}"
                else:
                    student_id = f"student_{i+1}"

                # Build answers dict for this student
                if has_subquestions and parsed_subquestions:
                    answers = parse_student_answers_from_text(
                        student_text, parsed_subquestions
                    )
                else:
                    answers = {"overall": student_text}

                payload = {
                    "question": question_dict,
                    "few_shot": few_shot_list,
                    "student_response": {
                        "response_id": student_id,
                        "answers": answers,
                    },
                    "has_subquestions": has_subquestions,
                    "images": images_b64,
                    "discipline": discipline,
                }

#Call backend
                try:
                    res = requests.post(BACKEND_URL, json=payload, timeout=300)
                    if res.status_code != 200:
                        st.warning(f"Row {i}: backend error {res.status_code}")
                        result_json = None
                    else:
                        result_json = res.json()
                except Exception as e:
                    st.warning(f"Row {i}: request failed: {e}")
                    result_json = None

#Fill result columns
                if result_json is not None:
                    results_df.at[i, "total_score"] = result_json.get("total_score")
                    results_df.at[i, "confidence"] = result_json.get("confidence")
                    results_df.at[i, "rationale"] = result_json.get("rationale")

                    fb = result_json.get("feedback", {}) or {}
                    results_df.at[i, "feedback_overall"] = fb.get("_overall", "")

                    sub_scores = result_json.get("sub_scores", {}) or {}
                    for sid, s_val in sub_scores.items():
                        col_name = f"sub_{sid}"
                        if col_name not in results_df.columns:
                            results_df[col_name] = None
                        results_df.at[i, col_name] = s_val

                progress.progress((i + 1) / max(1, total_rows))
                status_text.text(f"Graded {i+1}/{total_rows} student responses")

#Store in session state
            st.session_state.batch_results_df = sanitize_df_for_streamlit(results_df)
            st.success("Batch grading completed.")

    except Exception as e:
        st.error(f"Failed to process batch CSV: {e}")


#Right panel
with col_right:
    st.subheader(":violet[Results]")

    result = st.session_state.grade_result

    if result is None:
        st.info("Run grading to see student results.")
    else:
        override = st.session_state.override_mode

        total_score = result.get("total_score", 0.0)
        sub_scores = result.get("sub_scores", {})
        rationale = result.get("rationale", "")
        feedback = result.get("feedback", {})
        confidence = result.get("confidence", 0.0)

        #Total score front
        if override:
            total_score = st.number_input(
                "Total score",
                value=float(total_score),
                step=0.5,
                key="override_total_score",
            )
        else:
            st.metric("Total score", f"{total_score}")

        #subscore front
        st.markdown("### Sub-scores")
        if not has_subquestions:
            st.write("Sub-scores: NA (no subquestions enabled)")
        else:
            if isinstance(sub_scores, dict):
                for sid in sub_scores.keys():
                    val = sub_scores[sid]
                    if override:
                        try:
                            default_val = float(val)
                        except Exception:
                            default_val = 0.0
                        sub_scores[sid] = st.number_input(
                            f"Sub-score ({sid})",
                            value=default_val,
                            step=0.5,
                            key=f"override_subscore_{sid}",
                        )
                    else:
                        st.write(f"({sid}): {val}")
            else:
                st.write("Invalid sub_scores format from backend")

        #Rationale front
        st.markdown("### Rationale / Reasoning")
        if override:
            rationale = st.text_area(
                "Rationale", value=rationale, height=100, key="override_rationale"
            )
        else:
            st.write(rationale)

        #Feedback and improvement front
        st.markdown("### Feedback & Areas for improvement")
        if override:
            feedback_text = st.text_area(
                "Feedback (JSON)",
                value=json.dumps(feedback, indent=2),
                height=150,
                key="override_feedback",
            )
            try:
                feedback = json.loads(feedback_text)
            except Exception:
                st.error("Invalid feedback JSON; keeping original.")
        else:
            overall_fb = feedback.get("_overall")
            if overall_fb:
                st.write(f"Overall: {overall_fb}")
            for k, v in feedback.items():
                if k == "_overall":
                    continue
                st.write(f"({k}) {v}")

        #Confidence slide bar
        st.markdown("### Grading Confidence")
        if override:
            confidence = st.slider(
                "Confidence",
                min_value=0.0,
                max_value=100.0,
                value=float(confidence),
                step=0.05,
                key="override_confidence",
            )
        else:
            st.progress(min(1.0, max(0.0, confidence / 100.0)))
            st.write(f"{confidence:.1f}% confidence in this grading")

        #Human overide information items are saved after teacher edits
        if override:
            result["total_score"] = total_score
            result["sub_scores"] = sub_scores
            result["rationale"] = rationale
            result["feedback"] = feedback
            result["confidence"] = confidence
            st.session_state.grade_result = result
    
    st.markdown("---")
    st.subheader(":violet[Batch results]")

    if st.session_state.batch_results_df is not None:
        results_df = sanitize_df_for_streamlit(st.session_state.batch_results_df)

        st.dataframe(results_df, use_container_width=True)

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            ":green[Download batch graded results as CSV]",
            data=csv_data,
            file_name="voxlm_batch_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a CSV and click 'Grade uploaded CSV' in the SIDE BAR to see batch results here.")

#sidebar 2
with st.sidebar:
    prompt_text=""
    if st.session_state.grade_result is not None:
        prompt_text=st.session_state.grade_result.get("debug_prompt","") or ""
    
    st.text_area(
        "Full prompt sent to Vox-LM",
        value=prompt_text,
        height=400,
        key="debug_prompt_view"
    )
    st.caption(
        "Note: Editing this box does NOT change the prompt sent to the model. It is for debugging and inspection only."
    )
