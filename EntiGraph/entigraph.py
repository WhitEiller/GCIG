import argparse
import asyncio
import json
import os
import random
from hashlib import md5

from tqdm.asyncio import tqdm as tqdm_async

from baselines.EntiGraph.inference.devapi import gptqa
from baselines.EntiGraph.tasks.baseline_task import BaselineTask


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


async def generate_entities(
    document_content: str, system_message: str, openai_model: str
):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    can_read_entities = None

    max_tries = 5
    while not can_read_entities and max_tries > 0:
        try:
            completion = await gptqa(
                prompt, openai_model, system_message, json_format=False
            )
            completion = completion[completion.find("{") : completion.rfind("}") + 1]
            response = json.loads(completion)
            can_read_entities = response["entities"]
            return response
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to generate entities: {str(e)}")
            max_tries -= 1


async def generate_two_entity_relations(
    document_content: str,
    entity1: str,
    entity2: str,
    system_message: str,
    openai_model: str,
):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    """
    completion = await gptqa(prompt, openai_model, system_message)
    return completion


async def generate_three_entity_relations(
    document_content: str,
    entity1: str,
    entity2: str,
    entity3: str,
    system_message: str,
    openai_model: str,
):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    - {entity3}
    """
    completion = await gptqa(prompt, openai_model, system_message)
    return completion


def _post_process_synthetic_data(data):
    block = data.split("\n\n")
    qas = {}
    for line in block:
        if "Question: " in line and "Answer: " in line:
            question = line.split("Question: ")[1].split("Answer: ")[0]
            answer = line.split("Answer: ")[1]
            qas[compute_content_hash(question)] = {
                "question": question,
                "answer": answer,
            }
        break
    return qas


async def generate_synthetic_data_for_document(input_file, data_type, target_count=10000):
    random.seed(42)
    model_name = os.getenv("SYNTHESIZER_MODEL")
    task = BaselineTask(input_file, data_type)

    # Limit documents to process based on target_count
    # Estimate: each document generates ~1 summary + some entity pairs
    # Use 1.2x target_count as safety margin
    max_docs = int(target_count * 1.2)
    if len(task.documents) > max_docs:
        print(f"Limiting documents from {len(task.documents)} to {max_docs} (target: {target_count})")
        task.documents = random.sample(task.documents, max_docs)

    max_concurrent = 100
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_document_entities(doc):
        async with semaphore:
            try:
                entities = await generate_entities(
                    doc.text, task.openai_system_generate_entities, model_name
                )
                if not entities:
                    return None
                return {
                    "document": doc.text,
                    "entities": entities["entities"],
                    "summary": entities["summary"],
                }
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error generating entities: {str(e)[:100]}")
                return None

    entities_list = []
    for result in tqdm_async(
        asyncio.as_completed(
            [generate_document_entities(doc) for doc in task.documents]
        ),
        total=len(task.documents),
        desc="Generating entities",
    ):
        result = await result
        if result:
            entities_list.append(result)

    # iterate over triples of entities and generate relations
    # Calculate pairs per document to reach target_count
    # Each corpus item typically generates 1 QA pair, plus summaries from entities_list
    total_summaries = len(entities_list)
    target_pairs = max(target_count - total_summaries, 0)

    print(f"\n=== Generation Plan ===")
    print(f"Total documents processed: {len(entities_list)}")
    print(f"Target QA pairs: {target_count}")
    print(f"Expected from summaries: {total_summaries}")
    print(f"Expected from entity relations: {target_pairs}")

    pair_list = []
    for doc in entities_list:
        entities = doc["entities"]
        temp = []
        for i, entity_i in enumerate(entities):
            if i == len(entities) - 1:
                break
            for j in range(i + 1, len(entities)):
                entity_j = entities[j]
                pair = (doc["document"], entity_i, entity_j)
                temp.append(pair)

        # Dynamically calculate how many pairs to sample per document
        if len(entities_list) > 0:
            pairs_per_doc = max(1, target_pairs // len(entities_list))
        else:
            pairs_per_doc = 10
        pair_list.extend(random.sample(temp, min(len(temp), pairs_per_doc)))

    print(f"Total entity pairs to process: {len(pair_list)}")
    print(f"======================\n")

    async def process_two_entity_relations(pair):
        async with semaphore:
            try:
                document, entity1, entity2 = pair
                response = await generate_two_entity_relations(
                    document,
                    entity1,
                    entity2,
                    task.openai_system_generate_two_entity_relations,
                    model_name,
                )
                return response
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error generating relations: {str(e)[:100]}")
                return None

    corpus = []
    for result in tqdm_async(
        asyncio.as_completed(
            [process_two_entity_relations(pair) for pair in pair_list]
        ),
        total=len(pair_list),
        desc="Generating two entity relations",
    ):
        result = await result
        if result:
            corpus.append(result)

    # triple_list = []
    # for doc in entities_list:
    #     entities = doc['entities']
    #     for i in range(len(entities)):
    #         for j in range(i + 1, len(entities)):
    #             for k in range(j + 1, len(entities)):
    #                 triple = (doc['document'], entities[i], entities[j], entities[k])
    #                 triple_list.append(triple)
    #
    # async def process_three_entity_relations(triple):
    #     async with semaphore:
    #         document, entity1, entity2, entity3 = triple
    #         response = await generate_three_entity_relations(
    #             document, entity1, entity2, entity3,
    #             task.openai_system_generate_three_entity_relations,
    #             model_name)
    #         return response
    #
    # for result in tqdm_async(
    #         asyncio.as_completed([process_three_entity_relations(triple) for triple in triple_list]),
    #         total=len(triple_list),
    #         desc="Generating three entity relations"
    # ):
    #     corpus.append(await result)

    corpus = [doc["summary"] for doc in entities_list] + corpus

    # Limit corpus size to target_count with a small buffer
    max_corpus_size = int(target_count * 1.1)
    if len(corpus) > max_corpus_size:
        print(f"Limiting corpus from {len(corpus)} to {max_corpus_size} items")
        corpus = random.sample(corpus, max_corpus_size)

    qa_sft_results = {}

    async def generate_qa_sft(content):
        async with semaphore:
            completion = await gptqa(
                content, model_name, task.openai_system_quality_qa_sft
            )
            return completion

    # Create tasks so we can cancel them if needed
    tasks = [asyncio.create_task(generate_qa_sft(content)) for content in corpus]

    for result in tqdm_async(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Generating QA SFT",
    ):
        try:
            result = await result
            if result:
                qa_sft_results.update(_post_process_synthetic_data(result))
                # Stop if we've reached the target
                if len(qa_sft_results) >= target_count:
                    print(f"\nReached target count of {target_count}, stopping QA generation...")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
        except asyncio.CancelledError:
            pass  # Task was cancelled, ignore
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing QA: {str(e)[:100]}")

    print(f"\n=== Final Statistics ===")
    print(f"Total QA pairs generated: {len(qa_sft_results)}")
    print(f"========================\n")

    return qa_sft_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Raw context jsonl path.",
        default="/mnt/disk/yh24/test1/graphrag-purity/HotpotEval_Corpus.json",
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
        default="cache/data/entigraph.json",
        type=str,
    )
    parser.add_argument(
        "--target_count",
        help="Target number of QA pairs to generate.",
        default=10000,
        type=int,
    )

    args = parser.parse_args()

    try:
        results = asyncio.run(
            generate_synthetic_data_for_document(args.input_file, args.data_type, args.target_count)
        )

        print(f"\nGenerated {len(results)} QA pairs (target: {args.target_count})")

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
