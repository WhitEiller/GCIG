{
    "config_name": "bio_qa",
    "dataset_name": "BatsResearch/bonito-experiment-eval",
    "templates": {
        "Generate answer from context": {
            "jinja": 'Context: {{context}}\n\nQuestion: {{question}}\n\nAnswer: ||| {{answer}}',
            "answer_choices": None,
            "reference": "Generation format for bio_qa dataset",
        },
    },
}