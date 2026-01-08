import os
import sys

sys.path.append('//test1/graphrag-purity')

data_dir = "//test1/graphrag-purity/in_put"
question_path = os.path.join(data_dir, "MultiHopRAG.json")
corpus_path = os.path.join(data_dir, "corpus.json")  # TODO: 未合并的原始文本doc1, doc2, ...路径
text_path = os.path.join(data_dir, "text.txt")  # TODO: 合并后的原始的文本，将文档doc1, doc2, ...合并为merge_doc.txt

db_dir = "//test1/graphrag-purity/graph_zh"  # TODO: 输出路径
text_unit_dir = os.path.join(db_dir, "text_units")
chunk_dir = os.path.join(db_dir, "chunks")
graph_dir = os.path.join(db_dir, "graph")

results_dir = "results"

class BaseConfig:
    @classmethod
    def get_params(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

class SplitterConfig(BaseConfig):
    chunk_size = 2048
    over_lap = 64
    encoding_name = "cl100k_base"


class LLMConfig(BaseConfig):
    # TODO: 兼容openai库的大模型调用接口
    # model = "DeepSeek-V3"
    # api_key = os.environ.get("sk-52sOkyhq1y9R6mrn1VeHHXxFs9PbDrIF")
    # base_url = "http://10.4.177.241:9997/v1"
    model = "Qwen3-Coder"
    api_key = "134"
    base_url="http://10.4.177.69:9997/v1"


class ConnectorConfig(BaseConfig):
    expand = 1
    overlap_threshold = 0.9


class SimilarityAlignConfig(BaseConfig):
    threshold = 0.9