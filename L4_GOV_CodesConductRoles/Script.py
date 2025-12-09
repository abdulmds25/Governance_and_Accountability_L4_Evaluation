#!/usr/bin/env python3



import openai
import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

# Import for different file formats
import PyPDF2
from docx import Document


# ============================================================================
# FILE READING FUNCTIONS
# ============================================================================

def read_pdf(file_path: Path) -> Optional[str]:
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"  Reading {num_pages} pages...")
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        return text.strip()
    except Exception as e:
        print(f"  Error reading PDF: {e}")
        return None


def read_docx(file_path: Path) -> Optional[str]:
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"  Error reading DOCX: {e}")
        return None


def read_txt(file_path: Path) -> Optional[str]:
    """Read text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            print(f"  Error reading TXT: {e}")
            return None
    except Exception as e:
        print(f"  Error reading TXT: {e}")
        return None


def read_model_card(file_path: Path) -> Optional[str]:
    """
    Read model card from various file formats.
    Supports: .pdf, .txt, .doc, .docx
    """
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return None
    
    file_extension = file_path.suffix.lower()
    
    print(f"Reading {file_extension} file: {file_path.name}")
    
    if file_extension == '.pdf':
        return read_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return read_docx(file_path)
    elif file_extension == '.txt':
        return read_txt(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        print("Supported formats: .pdf, .txt, .doc, .docx")
        return None


# ============================================================================
# EVALUATION PROMPT
# ============================================================================

EVALUATION_PROMPT = """You are an AI governance and organizational responsibility expert. Evaluate the following documentation against the Governance Level 4 (L4_GOV_CodesConductRoles) standard.

**Governance L4 Standard Definition:**
Codes of conduct, roles, and training for AI responsibility are in place. This includes:
1. Published code of conduct or ethics guidelines for AI development and deployment
2. Clearly defined roles and responsibilities for AI governance
3. Training programs or requirements for staff working with AI systems
4. Evidence of implementation and enforcement mechanisms
5. Documentation of how these elements work together to ensure responsible AI practices

**Evaluation Framework:**

For each criterion below, provide:
- Score: 0 (Not Met), 1 (Partially Met), 2 (Fully Met)
- Evidence: Specific quotes or observations from the document
- Gaps: What's missing or could be improved

**Criteria to Evaluate:**

1. **Code of Conduct / Ethics Guidelines**
   - Explicit code of conduct for AI development published
   - Clear ethical principles or values stated
   - Guidelines cover key areas (safety, fairness, privacy, accountability)
   - Scope clearly defined (who must follow, which activities covered)
   - Consequences for violations mentioned

2. **Defined Roles and Responsibilities**
   - Specific AI governance roles identified (e.g., AI Ethics Board, Chief AI Officer, Safety Team)
   - Clear accountability structure established
   - Decision-making authority specified
   - Reporting relationships documented
   - Cross-functional coordination described

3. **Training Programs**
   - Training requirements for AI practitioners documented
   - Training content described (responsible AI, ethics, safety practices)
   - Target audiences specified (engineers, leadership, all staff)
   - Frequency or ongoing nature of training mentioned
   - Assessment or certification components included

4. **Implementation Evidence**
   - Examples of policies being applied in practice
   - Process documentation (how decisions are made, how issues are escalated)
   - Integration with development lifecycle described
   - Monitoring or compliance mechanisms mentioned
   - Incident response procedures documented

5. **Enforcement and Accountability**
   - Mechanisms to ensure compliance described
   - Consequences for violations specified
   - Auditing or review processes mentioned
   - Whistleblower or reporting mechanisms included
   - External oversight or validation referenced

6. **Integration and Coherence**
   - Clear connections between codes, roles, and training
   - Comprehensive coverage across the AI lifecycle
   - Regular review and update processes described
   - Alignment with external standards or frameworks mentioned
   - Accessibility and discoverability of documentation

**Output Format:**

Provide your evaluation in the following JSON structure:
```json
{{
  "overall_score": "X/12",
  "overall_rating": "Does Not Meet L4 / Partially Meets L4 / Fully Meets L4",
  "criteria_scores": {{
    "code_of_conduct": {{"score": X, "evidence": "...", "gaps": "..."}},
    "roles_responsibilities": {{"score": X, "evidence": "...", "gaps": "..."}},
    "training_programs": {{"score": X, "evidence": "...", "gaps": "..."}},
    "implementation_evidence": {{"score": X, "evidence": "...", "gaps": "..."}},
    "enforcement_accountability": {{"score": X, "evidence": "...", "gaps": "..."}},
    "integration_coherence": {{"score": X, "evidence": "...", "gaps": "..."}}
  }},
  "strengths": ["strength 1", "strength 2", "..."],
  "weaknesses": ["weakness 1", "weakness 2", "..."],
  "recommendations": ["recommendation 1", "recommendation 2", "..."],
  "summary": "2-3 sentence overall assessment"
}}
```

**Documentation to Evaluate:**

{model_card_content}

Provide a thorough, evidence-based evaluation."""


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model_card(model_card_content: str, model: str = "gpt-4o", 
                       temperature: float = 0.2) -> Optional[Dict]:
    """
    Evaluate a model card using OpenAI's API.
    
    Args:
        model_card_content: The full text of the model card
        model: OpenAI model to use
        temperature: Temperature for generation
    
    Returns:
        Evaluation results dictionary
    """
    try:
        full_prompt = EVALUATION_PROMPT.format(model_card_content=model_card_content)
        
        print(f"  Calling OpenAI API ({model})...")
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI transparency and governance expert specializing in evaluating AI model documentation against industry standards."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            max_tokens=4000
        )
        
        evaluation_text = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            if "```json" in evaluation_text:
                json_start = evaluation_text.find("```json") + 7
                json_end = evaluation_text.find("```", json_start)
                json_str = evaluation_text[json_start:json_end].strip()
                evaluation_result = json.loads(json_str)
            else:
                evaluation_result = json.loads(evaluation_text)
        except json.JSONDecodeError:
            evaluation_result = {
                "raw_evaluation": evaluation_text,
                "note": "Could not parse as JSON, returning raw text"
            }
        
        return evaluation_result
        
    except Exception as e:
        print(f"  Error during API call: {e}")
        return None


def save_evaluation(evaluation_result: Dict, output_file: Path) -> None:
    """Save evaluation results to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved to {output_file}")
    except Exception as e:
        print(f"  Error saving evaluation: {e}")


def print_summary(evaluation_result: Dict, file_name: str = "") -> None:
    """Print a formatted summary of the evaluation."""
    if "raw_evaluation" in evaluation_result:
        print("\n" + "="*80)
        print("EVALUATION RESULT")
        print("="*80)
        print(evaluation_result["raw_evaluation"])
        return
    
    print("\n" + "="*80)
    if file_name:
        print(f"EVALUATION SUMMARY: {file_name}")
    else:
        print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Overall Score: {evaluation_result.get('overall_score', 'N/A')}")
    print(f"📈 Overall Rating: {evaluation_result.get('overall_rating', 'N/A')}")
    
    print("\n" + "-"*80)
    print("CRITERIA SCORES")
    print("-"*80)
    for criterion, details in evaluation_result.get('criteria_scores', {}).items():
        score = details.get('score', 0)
        status = "✅" if score == 2 else ("⚠️ " if score == 1 else "❌")
        print(f"\n{status} {criterion.replace('_', ' ').title()}: {score}/2")
        if details.get('gaps'):
            print(f"   Gap: {details.get('gaps', 'N/A')[:100]}...")
    
    print("\n" + "-"*80)
    print("STRENGTHS")
    print("-"*80)
    for strength in evaluation_result.get('strengths', []):
        print(f"  ✓ {strength}")
    
    print("\n" + "-"*80)
    print("WEAKNESSES")
    print("-"*80)
    for weakness in evaluation_result.get('weaknesses', []):
        print(f"  ✗ {weakness}")
    
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)
    for rec in evaluation_result.get('recommendations', []):
        print(f"  → {rec}")
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    print(evaluation_result.get('summary', 'N/A'))
    print("\n" + "="*80 + "\n")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_file(file_path: Path, model: str, temperature: float, 
                output_dir: Path, verbose: bool = True) -> Optional[Dict]:
    """Process a single file."""
    print(f"\n{'='*80}")
    print(f"Processing: {file_path.name}")
    print('='*80)
    
    # Read file
    content = read_model_card(file_path)
    if not content:
        print(f"  ✗ Failed to read file")
        return None
    
    print(f"  ✓ Loaded {len(content):,} characters, {len(content.split()):,} words")
    
    # Evaluate
    result = evaluate_model_card(content, model, temperature)
    if not result:
        print(f"  ✗ Evaluation failed")
        return None
    
    print(f"  ✓ Evaluation complete")
    
    # Save
    output_file = output_dir / f"{file_path.stem}_evaluation.json"
    save_evaluation(result, output_file)
    
    # Print summary
    if verbose:
        print_summary(result, file_path.name)
    else:
        score = result.get('overall_score', 'N/A')
        rating = result.get('overall_rating', 'N/A')
        print(f"  Score: {score} | Rating: {rating}")
    
    return result


def batch_process(file_paths: List[Path], model: str, temperature: float, 
                 output_dir: Path) -> Dict[str, Dict]:
    """Process multiple files."""
    results = {}
    
    print(f"\n{'#'*80}")
    print(f"BATCH PROCESSING: {len(file_paths)} files")
    print('#'*80)
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}]")
        result = process_file(file_path, model, temperature, output_dir, verbose=False)
        if result:
            results[file_path.name] = result
    
    # Print comparison summary
    print(f"\n{'#'*80}")
    print("BATCH SUMMARY")
    print('#'*80)
    print(f"\n{'File':<40} {'Score':<10} {'Rating':<20}")
    print("-"*80)
    for filename, result in results.items():
        score = result.get('overall_score', 'N/A')
        rating = result.get('overall_rating', 'N/A')
        print(f"{filename:<40} {score:<10} {rating:<20}")
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI model cards against Transparency L4 standard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transparency_evaluator.py model_card.pdf
  python transparency_evaluator.py model_card.txt --model gpt-4-turbo
  python transparency_evaluator.py *.pdf --batch
  python transparency_evaluator.py file1.pdf file2.docx --batch --output results/
        """
    )
    
    parser.add_argument('files', nargs='+', help='Model card file(s) to evaluate')
    parser.add_argument('--model', default='gpt-4o', 
                       choices=['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for generation (default: 0.2)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--output', '-o', default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: process multiple files with summary')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode: minimal output')
    
    args = parser.parse_args()
    
    # Setup API key
    if args.api_key:
        openai.api_key = args.api_key
    elif not openai.api_key:
        print("Error: OpenAI API key not found.")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert file arguments to Path objects
    file_paths = [Path(f) for f in args.files]
    
    # Validate files exist
    missing_files = [f for f in file_paths if not f.exists()]
    if missing_files:
        print("Error: The following files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    # Process files
    try:
        if len(file_paths) > 1 or args.batch:
            batch_process(file_paths, args.model, args.temperature, output_dir)
        else:
            process_file(file_paths[0], args.model, args.temperature, 
                        output_dir, verbose=not args.quiet)
        
        print("\n✓ All evaluations complete!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
