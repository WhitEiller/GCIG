path = "question_outputs_model4"

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def find_answer_section(lines: List[str]) -> int:
    """Return the index of the line after the '**Answer**' marker. -1 if not found."""
    for idx, line in enumerate(lines):
        if "**Answer**" in line:
            return idx + 1
    return -1


def find_and_parse_sources(lines: List[str]) -> str:
    """Find and parse the **Sources** section, returning the source text."""
    for idx, line in enumerate(lines):
        if "**Sources**" in line:
            if idx + 1 < len(lines):
                source_line = lines[idx + 1].strip()
                if source_line.startswith('[') and source_line.endswith(']'):
                    try:
                        source_list = eval(source_line)
                        if isinstance(source_list, list) and len(source_list) > 0:
                            return ' '.join(source_list)
                    except:
                        pass
                return source_line
    return ""


def parse_qa_from_lines(lines: List[str]) -> List[Tuple[str, str]]:
    """Parse up to three (question, answer) pairs from lines after the Answer marker.

    Expected pattern per pair (spacing tolerant):
    - ***Question N***: <question text>
    - ***Answer N***: <answer text>
    """
    qa_pairs: List[Tuple[str, str]] = []
    # Regexes tolerate extra spaces and optional trailing spaces
    q_re = re.compile(r"^\*\*\*Question\s+(\d+)\*\*\*:\s*(.+?)\s*$")
    a_re = re.compile(r"^\*\*\*Answer\s+(\d+)\*\*\*:\s*(.+?)\s*$")

    i = 0
    while i < len(lines):
        q_match = q_re.match(lines[i])
        if q_match:
            q_num = q_match.group(1)
            question_text = q_match.group(2).strip()

            # If question text is empty on the header line, take the next non-empty line before the answer marker
            j = i + 1
            if not question_text:
                k = j
                while k < len(lines):
                    if a_re.match(lines[k]) or q_re.match(lines[k]):
                        break
                    if lines[k].strip():
                        question_text = lines[k].strip()
                        break
                    k += 1

            # Look ahead for matching answer line
            while j < len(lines) and not a_re.match(lines[j]):
                j += 1
            if j < len(lines):
                a_match = a_re.match(lines[j])
                if a_match and a_match.group(1) == q_num:
                    answer_text = a_match.group(2).strip()
                    # If answer text is empty on the header line, take the next non-empty line
                    if not answer_text and j + 1 < len(lines):
                        k = j + 1
                        while k < len(lines):
                            if q_re.match(lines[k]) or a_re.match(lines[k]):
                                break
                            if lines[k].strip():
                                answer_text = lines[k].strip()
                                break
                            k += 1
                    qa_pairs.append((question_text, answer_text))
                    i = j + 1
                    continue
        i += 1
    # Only keep first three if more were found
    return qa_pairs[:3]


def process_txt_file(txt_path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """Process a txt file and return (source, qa_pairs)."""
    content = txt_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.rstrip("\n\r") for line in content.splitlines()]
    
    source = find_and_parse_sources(lines)
    
    start = find_answer_section(lines)
    if start == -1:
        return source, []
    return source, parse_qa_from_lines(lines[start:])


def main() -> None:
    base_dir = Path(__file__).parent
    input_dir = (base_dir / path).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    per_file_map: Dict[str, Dict[str, any]] = {}
    sft_lines: List[Dict[str, str]] = []

    for txt_file in sorted(input_dir.glob("*.txt")):
        source, qa_list = process_txt_file(txt_file)
        per_file_map[txt_file.name] = {
            "source": source,
            "qa_pairs": [{"question": q, "answer": a} for (q, a) in qa_list]
        }
        for q, a in qa_list:
            sft_lines.append({"source": source, "question": q, "answer": a})

    # Outputs in current folder (graph_to_sft)
    extracted_path = base_dir / "extracted.json"
    sft_path = base_dir / f"{path}_sft.jsonl"

    # extracted_path.write_text(
    #     json.dumps(per_file_map, ensure_ascii=False, indent=2), encoding="utf-8"
    # )
    with sft_path.open("w", encoding="utf-8") as f:
        for item in sft_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Extracted {len(sft_lines)} QA pairs.")


if __name__ == "__main__":
    main()