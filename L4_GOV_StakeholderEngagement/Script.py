#!/usr/bin/env python3

import openai
import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import PyPDF2
from docx import Document


# ============================================================================
# FILE READING FUNCTIONS
# ============================================================================

def read_pdf(file_path: Path) -> Optional[str]:
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"  Reading {len(reader.pages)} pages...")

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        return text.strip()
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return None


def read_docx(file_path: Path) -> Optional[str]:
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        print(f"[DOCX ERROR] {e}")
        return None


def read_txt(file_path: Path) -> Optional[str]:
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except:
        return file_path.read_text(encoding="latin-1").strip()


def read_model_card(file_path: Path) -> Optional[str]:
    ext = file_path.suffix.lower()
    print(f"Reading {file_path.name} ({ext})")

    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return read_docx(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return None


# ============================================================================
# CHUNKING (WITH OVERLAP)
# ============================================================================

def chunk_text(text: str, chunk_size=12000, overlap=300) -> List[str]:
    """Split text into overlapping chunks so GPT does not lose context."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # maintain context continuity

    return chunks


# ============================================================================
# L4 PROMPT TEMPLATE
# ============================================================================

EVALUATION_PROMPT = """
You are an AI governance and stakeholder engagement expert. Evaluate the following document
STRICTLY for the Governance L4 sub-dimension: **L4_GOV_StakeholderEngagement**.

**Standard Definition:**
Stakeholder engagement and escalation paths are defined. This includes:
1. Clear identification of relevant stakeholders (internal and external)
2. Documented engagement processes and mechanisms for stakeholder input
3. Defined escalation paths for issues, concerns, and decision-making
4. Evidence of active stakeholder participation in AI governance
5. Transparency about how stakeholder feedback influences decisions

Return ONLY VALID JSON following the required schema. NO prose. NO markdown.

**Required JSON Schema:**
{{
  "overall_score": "X/12",
  "overall_rating": "Does Not Meet L4 / Partially Meets L4 / Fully Meets L4",
  "criteria_scores": {{
    "stakeholder_identification": {{"score": X, "evidence": "...", "gaps": "..."}},
    "engagement_mechanisms": {{"score": X, "evidence": "...", "gaps": "..."}},
    "escalation_paths": {{"score": X, "evidence": "...", "gaps": "..."}},
    "active_participation": {{"score": X, "evidence": "...", "gaps": "..."}},
    "transparency_feedback": {{"score": X, "evidence": "...", "gaps": "..."}},
    "coverage_accessibility": {{"score": X, "evidence": "...", "gaps": "..."}}
  }},
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "summary": "2-3 sentence overall assessment"
}}

DOCUMENT CONTENT:
{model_card_content}
"""


# ============================================================================
# FORCED JSON EVALUATION
# ============================================================================

def evaluate_model_card(text: str, model="gpt-4o", temperature=0.2):
    """Force GPT to ALWAYS output valid JSON even if input is messy."""

    system_override = (
        "You MUST ALWAYS return ONLY valid JSON following the schema. "
        "If the document resembles a governance report, policy document, "
        "stakeholder engagement plan, or incomplete text, YOU STILL MUST evaluate it. "
        "NEVER refuse. NEVER output explanations. NEVER say what the text looks like. "
        "Provide your best JSON evaluation even if evidence is limited. "
        "Score each criterion 0-2: 0=Not Met, 1=Partially Met, 2=Fully Met."
    )

    try:
        full_prompt = EVALUATION_PROMPT.format(model_card_content=text)

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_override},
                {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            max_tokens=4000
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON safely
        if "{" in raw and "}" in raw:
            try:
                json_str = raw[raw.find("{"): raw.rfind("}") + 1]
                return json.loads(json_str)
            except Exception as e:
                return {"error": "JSON parse failed", "raw_output": raw, "exception": str(e)}

        return {"error": "No JSON detected", "raw_output": raw}

    except Exception as e:
        print("[API ERROR]", e)
        return None


# ============================================================================
# MERGE ALL CHUNKS USING MAX SCORING
# ============================================================================

def merge_results(results: List[Dict]) -> Dict:
    """Merge chunk evaluations using MAX score across all chunks."""

    criteria = [
        "stakeholder_identification",
        "engagement_mechanisms",
        "escalation_paths",
        "active_participation",
        "transparency_feedback",
        "coverage_accessibility"
    ]

    merged = {
        "overall_score": None,
        "overall_rating": None,
        "criteria_scores": {},
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
        "summary": ""
    }

    # Initialize default scores
    for c in criteria:
        merged["criteria_scores"][c] = {
            "score": 0,
            "evidence": [],
            "gaps": []
        }

    valid_chunks = 0

    for r in results:
        if not r or "criteria_scores" not in r:
            continue

        valid_chunks += 1

        for c in criteria:
            cdata = r["criteria_scores"].get(c, {})
            score = cdata.get("score", 0)

            # MAX rule: take highest score across all chunks
            if score > merged["criteria_scores"][c]["score"]:
                merged["criteria_scores"][c]["score"] = score

            merged["criteria_scores"][c]["evidence"].append(cdata.get("evidence", ""))
            merged["criteria_scores"][c]["gaps"].append(cdata.get("gaps", ""))

        merged["strengths"] += r.get("strengths", [])
        merged["weaknesses"] += r.get("weaknesses", [])
        merged["recommendations"] += r.get("recommendations", [])

    if valid_chunks == 0:
        merged["overall_score"] = "0/12"
        merged["overall_rating"] = "Does Not Meet L4"
        merged["summary"] = "No valid chunk evaluations found."
        return merged

    # Total score = sum of max scores (0–12)
    total = sum(merged["criteria_scores"][c]["score"] for c in criteria)

    merged["overall_score"] = f"{total}/12"

    if total >= 10:
        merged["overall_rating"] = "Fully Meets L4"
    elif total >= 6:
        merged["overall_rating"] = "Partially Meets L4"
    else:
        merged["overall_rating"] = "Does Not Meet L4"

    # Deduplicate and clean up lists
    merged["strengths"] = list(set(merged["strengths"]))
    merged["weaknesses"] = list(set(merged["weaknesses"]))
    merged["recommendations"] = list(set(merged["recommendations"]))

    merged["summary"] = (
        f"Merged from {valid_chunks} chunks using MAX scoring per criterion. "
        f"Final overall score = {total}/12 for L4_GOV_StakeholderEngagement."
    )

    return merged


# ============================================================================
# PROCESS A FILE: CHUNK → EVAL → MERGE
# ============================================================================

def process_file(file_path: Path, model: str, temperature: float, output_dir: Path):

    content = read_model_card(file_path)
    if not content:
        print("ERROR: Could not read file.")
        return None

    chunks = chunk_text(content, chunk_size=12000, overlap=300)
    print(f"SUCCESS: Split into {len(chunks)} chunks")

    results = []

    # Run evaluation for each chunk
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n>>> Evaluating chunk {i}/{len(chunks)}...")
        r = evaluate_model_card(chunk, model=model, temperature=temperature)
        results.append(r)

        # Save chunk JSON
        chunk_path = output_dir / f"{file_path.stem}_chunk{i}.json"
        chunk_path.write_text(json.dumps(r, indent=2))
        print(f"    Saved: {chunk_path.name}")

    # Merge all chunks
    print("\n>>> Merging all chunk results...")
    merged = merge_results(results)

    merged_path = output_dir / f"{file_path.stem}_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2))

    print(f"\nSUCCESS: MERGED EVALUATION SAVED: {merged_path}\n")
    print_summary(merged)

    return merged


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_summary(evaluation_result: Dict) -> None:
    """Print a formatted summary of the evaluation."""
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY: L4_GOV_StakeholderEngagement")
    print("="*80)
    
    print(f"\nOverall Score: {evaluation_result.get('overall_score', 'N/A')}")
    print(f"Overall Rating: {evaluation_result.get('overall_rating', 'N/A')}")
    
    print("\n" + "-"*80)
    print("CRITERIA SCORES")
    print("-"*80)
    for criterion, details in evaluation_result.get('criteria_scores', {}).items():
        score = details.get('score', 0)
        status = "[PASS]" if score == 2 else ("[PARTIAL]" if score == 1 else "[FAIL]")
        print(f"\n{status} {criterion.replace('_', ' ').title()}: {score}/2")
        
        # Show first evidence if available
        evidence = details.get('evidence', [])
        if evidence and isinstance(evidence, list) and evidence[0]:
            print(f"   Evidence: {evidence[0][:100]}...")
        
        # Show first gap if available
        gaps = details.get('gaps', [])
        if gaps and isinstance(gaps, list) and gaps[0]:
            print(f"   Gap: {gaps[0][:100]}...")
    
    print("\n" + "-"*80)
    print("STRENGTHS")
    print("-"*80)
    for strength in evaluation_result.get('strengths', [])[:5]:  # Show top 5
        print(f"  + {strength}")
    
    print("\n" + "-"*80)
    print("WEAKNESSES")
    print("-"*80)
    for weakness in evaluation_result.get('weaknesses', [])[:5]:  # Show top 5
        print(f"  - {weakness}")
    
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)
    for rec in evaluation_result.get('recommendations', [])[:5]:  # Show top 5
        print(f"  > {rec}")
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    print(evaluation_result.get('summary', 'N/A'))
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI governance documentation for L4_GOV_StakeholderEngagement"
    )
    parser.add_argument("files", nargs="+", help="Documentation file(s) to evaluate")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--output", default=".", help="Output directory for results")
    args = parser.parse_args()

    if args.api_key:
        openai.api_key = args.api_key

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*80}")
    print("L4_GOV_STAKEHOLDER ENGAGEMENT EVALUATOR")
    print('#'*80)

    for f in args.files:
        print(f"\n{'='*80}")
        print(f"Processing: {f}")
        print('='*80)
        process_file(Path(f), args.model, args.temperature, output_dir)

    print("\nSUCCESS: All evaluations complete!")


if __name__ == "__main__":
    main()