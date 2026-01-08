import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from hashlib import md5
from typing import List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

from baselines.EntiGraph.inference.devapi import gptqa


def compute_content_hash(content, prefix: str = ""):
    if content is None:
        content = ""
    return prefix + md5(content.encode()).hexdigest()


def create_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

INSTRUCTION_GENERATION_PROMPT = """The background knowledge is:
{doc}

Please generate one instruction question based on the content of the above article.
The question can be a question about facts or an understanding and evaluation of relevant content.
Please assume that there is no corresponding article to refer to when asking questions, so do not use demonstrative pronouns such as "this" or "these" in the question.

Please generate the question in the following format:
Question: ...
"""

READING_COMPREHENSION_PROMPT = """The background knowledge is:
{doc}
Please answer the following question based on the content of the article above:
{question}

Please answer this question as thoroughly as possible, but do not change the key information in the original text, and do not include expressions such as "based on the above article" in the answer.

Please generate the corresponding answer in the following format:
Question: ...
Answer: ...
"""


def _post_process_instructions(content: str) -> list:
    if not content:
        return []
    lines = content.split("\n")
    questions = []
    for line in lines:
        if "Question:" in line:
            question = line.split("Question:")[1].strip()
            questions.append(question)
    return questions


def _post_process_answers(content: str) -> tuple:
    if not content:
        return None, None
    if "Question:" in content and "Answer:" in content:
        question = content.split("Question:")[1].split("Answer:")[0].strip()
        answer = content.split("Answer:")[1].strip()
        return question, answer
    return None, None


@dataclass
class SelfQA:
    model_name: str = None
    max_concurrent: int = 100

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> dict:
        final_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_chunk(content: str):
            async with semaphore:
                prompt = INSTRUCTION_GENERATION_PROMPT.format(doc=content)
                try:
                    response = await gptqa(prompt, self.model_name, "You are a helpful assistant.")
                    if not response:
                        print("Warning: LLM returned empty response")
                        return []
                    instruction_questions = _post_process_instructions(response)

                    qas = []
                    for qa in tqdm_async(
                        asyncio.as_completed(
                            [
                                gptqa(
                                    READING_COMPREHENSION_PROMPT.format(
                                        doc=content, question=question
                                    ),
                                    self.model_name,
                                    "You are a helpful assistant."
                                )
                                for question in instruction_questions
                            ]
                        ),
                        total=len(instruction_questions),
                        desc="Generating QAs",
                    ):
                        try:
                            qa_response = await qa
                            if not qa_response:
                                continue
                            question, answer = _post_process_answers(qa_response)
                            if question and answer:
                                qas.append(
                                    {
                                        compute_content_hash(question): {
                                            "question": question,
                                            "answer": answer,
                                        }
                                    }
                                )
                        except Exception as e:  # pylint: disable=broad-except
                            print(f"Error processing QA: {e}")
                            continue
                    return qas
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error: {e}")
                    return []

        tasks = []
        for doc in docs:
            for chunk in doc:
                try:
                    if isinstance(chunk, dict) and 'content' in chunk:
                        # Clean content from null bytes
                        content = chunk["content"].replace('\x00', '')
                        if content.strip():  # Only process non-empty content
                            tasks.append(process_chunk(content))
                except Exception as e:
                    print(f"Warning: Skipping invalid chunk: {str(e)[:100]}")
                    continue

        for result in tqdm_async(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating using SelfQA",
        ):
            try:
                qas = await result
                for qa in qas:
                    final_results.update(qa)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error: {e}")
        return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Raw context jsonl path.",
        default="//test1/graphrag-purity/HotpotEval_Corpus.json",
        type=str,
    )
    parser.add_argument(
        "--data_type",
        help="Data type of input file. (Raw context or chunked context)",
        choices=["raw", "chunked"],
        default="chunked",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="Output file path.",
        default="cache/data/self-qa.json",
        type=str,
    )
    parser.add_argument(
        "--target_count",
        help="Target number of QA pairs to generate (0 means process all documents).",
        default=10000,
        type=int,
    )

    args = parser.parse_args()

    load_dotenv()

    model_name = os.getenv("SYNTHESIZER_MODEL")

    self_qa = SelfQA(model_name=model_name)

    try:
        # Load data with error handling
        if args.data_type == "raw":
            with open(args.input_file, "r", encoding="utf-8", errors='ignore') as f:
                data = []
                for line in f:
                    try:
                        data.append(json.loads(line.replace('\x00', ''), strict=False))
                    except Exception as e:
                        print(f"Warning: Skipping invalid JSON line: {str(e)[:100]}")
                        continue
                data = [[chunk] for chunk in data]
        elif args.data_type == "chunked":
            with open(args.input_file, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read().replace('\x00', '')
                data = json.loads(content, strict=False)

        print(f"Loaded {len(data)} documents")

        # Limit data if target_count is specified and > 0
        if args.target_count > 0:
            # Each document generates 1 QA pair
            max_docs = args.target_count
            if len(data) > max_docs:
                print(f"Limiting to first {max_docs} documents to reach target of {args.target_count} QA pairs")
                data = data[:max_docs]

        results = self_qa.generate(data)

        print(f"\n=== Final Statistics ===")
        print(f"Total QA pairs generated: {len(results)}")
        print(f"Target was: {args.target_count}")
        print(f"========================\n")

        # Save results
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to: {args.output_file}")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
