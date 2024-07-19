from datasets import load_dataset
import cohere
from sentence_transformers.quantization import quantize_embeddings
from scipy import spatial
import time
from pympler import asizeof
import numpy as np
from typing import List
# Load dataset
# https://huggingface.co/datasets/wikimedia/wikipedia
docs_stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

# cohere token
api_key = ""
co = cohere.Client(api_key)

dataset_size = 1000
calibration_set_size = int(dataset_size/10)
queries_num = 10

numpy_types_dict = {
    "float": np.float32,
    "int8": np.int8,
    "uint8": np.uint8
}
def populate_texts_list(docs_stream, n_docs):
    texts = []
    start_time = time.time()
    for i, doc in enumerate(docs_stream):
        texts.append(doc['text'])
        # ids.append(doc['id'])
        if i == n_docs - 1:
            break
    total_time = time.time() - start_time
    assert len(texts) == n_docs
    return texts, total_time

def knn_L2(query, doc_embeddings, k = 10):
    # compute the euclidean distance between the query and all the documents
    res = [(spatial.distance.euclidean(query, vec), id) for id, vec in enumerate(doc_embeddings)]
    res = sorted(res)
    # return the top k documents
    return res[0:k]

RES_ID = 1
def count_correct(gt_results, results):
    # compute how many ids in res appear in GT
    correct = 0
    for res in results:
        for gt_res in gt_results:
            if res[RES_ID] == gt_res[RES_ID]:
                correct += 1
                break

    return correct

def dataset_and_queries_embeddings(dataset_texts, queries_texts, embedding_types: List[str]):
    dataset_and_queries_embeddings_dict = {}
    for embedding_type in embedding_types:
        start_time = time.time()
        dataset_embeddings = co.embed(texts=dataset_texts, model="embed-english-v3.0", input_type="search_document", embedding_types=[embedding_type]).embeddings
        dataset_time = time.time() - start_time
        dataset_embeddings = getattr(dataset_embeddings, embedding_type)
        # numpy has less memory overhead than a list
        dataset_embeddings = np.array(dataset_embeddings, dtype=numpy_types_dict[embedding_type])
        # dataset_embeddings = np.array(dataset_embeddings)


        print(f"Generating {len(dataset_embeddings)} dataset embeddings took {dataset_time} seconds")
        start_time = time.time()

        queries_embeddings = co.embed(texts=queries_texts, model="embed-english-v3.0", input_type="search_query", embedding_types=[embedding_type]).embeddings
        quries_time = time.time() - start_time
        queries_embeddings = getattr(queries_embeddings, embedding_type)
        queries_embeddings = np.array(queries_embeddings, dtype=numpy_types_dict[embedding_type])
        # queries_embeddings = np.array(queries_embeddings)
        print(f"Generating {len(queries_embeddings)} queries embeddings took {quries_time} seconds")
        dataset_and_queries_embeddings_dict[embedding_type] = (dataset_embeddings, queries_embeddings)
        return dataset_and_queries_embeddings_dict

def dataset_and_queries_SQ_embeddings(float_embeddings, float_queries_embeddings, embedding_type: str):
    start_time = time.time()
    dataset_sq_embeddings = quantize_embeddings(
        float_embeddings,
        precision=embedding_type,
        calibration_embeddings=float_embeddings[:calibration_set_size],)
    dataset_time = time.time() - start_time
    print(f"Quantizing {len(dataset_sq_embeddings)} dataset embeddings took {dataset_time} seconds")

    start_time = time.time()
    query_sq_embeddings = quantize_embeddings(
        float_queries_embeddings,
        precision=embedding_type,
        calibration_embeddings=float_embeddings[:calibration_set_size],)
    quries_time = time.time() - start_time
    print(f"Quantizing {len(query_sq_embeddings)} queries embeddings took {quries_time} seconds")
    return dataset_sq_embeddings, query_sq_embeddings

def batch_knn(queries_embeddings, dataset_embeddings, k = 10):
    res = []
    for query in queries_embeddings:
        res.append(knn_L2(query, dataset_embeddings, k = 10))
    return res


def size_in_mb_pympler(obj):
    return asizeof.asizeof(obj) / (1024 * 1024)

VECTORS=0
QUERIES=1
def main():
    k = 10
    dataset_texts, dataset_set_loadtime = populate_texts_list(docs_stream, dataset_size)
    print(f"Loaded dataset with {dataset_size} documents took {dataset_set_loadtime} seconds")

    dataset_size_mb_pympler = size_in_mb_pympler(dataset_texts)
    print(f"Memory of dataset_texts: {dataset_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M documents: {float(1_000_000/dataset_size) * (dataset_size_mb_pympler/1024)} GB")

    queries_texts, queries_loadtime = populate_texts_list(docs_stream, queries_num)
    print(f"Loaded {queries_num} queries with took {queries_loadtime} seconds")

    # Get GT ids
    # generate dataset embeddings using a dedicated model
    # dataset_and_queries_embeddings returns a dict of (type_embeddings, type_queries_embeddings) tuples
    embeddings_dict = dataset_and_queries_embeddings(dataset_texts, queries_texts, ["float", "int8", "uint8"])

    embed_size_mb_pympler = size_in_mb_pympler(embeddings_dict["float"][VECTORS])
    print(f"Memory of float_embeddings: {embed_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M documents: {float(1_000_000/dataset_size) * (embed_size_mb_pympler/1024)} GB")

    gt_res = batch_knn(embeddings_dict["float"][QUERIES], embeddings_dict["float"][VECTORS], k = 10)

    for onebyte_type in ["int8", "uint8"]:
        embeddings, queries_embeddings = embeddings_dict[onebyte_type]
        print(f"Generated {onebyte_type} embeddings. Example vec slice = {embeddings[0][:10]}")
        correct = 0
        for i, query in enumerate(queries_embeddings):
            res = knn_L2(query, embeddings, k = 10)
            correct += count_correct(gt_res[i], res)
        recall = correct / (k * queries_num)
        print(f"type = {onebyte_type}, embeddings from model\nRecall: ", recall)

        # generate dataset embeddings using a SQ
        sq_embeddings, sq_queries_embeddings = dataset_and_queries_SQ_embeddings(float_embeddings, float_query_embeddings, onebyte_type)
        print(f"Quantized {onebyte_type} embeddings. Example vec slice = {sq_embeddings[0][:10]}")

        correct = 0
        for i, query in enumerate(sq_queries_embeddings):
            res = knn_L2(query, sq_embeddings, k = 10)
            correct += count_correct(gt_res[i], res)
        recall = correct / (k * queries_num)
        print(f"type = {onebyte_type}, scalar quantization embeddings\nRecall: ", recall)


if __name__ == "__main__":
    main()
