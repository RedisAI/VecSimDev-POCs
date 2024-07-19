from tracemalloc import start
from datasets import load_dataset
import cohere
from sentence_transformers.quantization import quantize_embeddings
from scipy import spatial
import time
from pympler import asizeof
import numpy as np

# Load dataset

# Get embeddings
# https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary
lang = "simple" #Use the Simple English Wikipedia subset
int8_docs_stream = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", lang, split="train", streaming=True)
float32_docs_stream = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split="train", streaming=True)

dataset_size = 1000
calibration_set_size = int(dataset_size/10)
queries_num = 10


numpy_types_dict = {
    "float": np.float32,
    "int8": np.int8,
    "uint8": np.uint8
}

def dataset_and_queries_embeddings(docs_stream, embeddings_field, embedding_type):
    dataset_embeddings = []
    # dataset_titles = []
    start_time = time.time()
    for i, doc in enumerate(docs_stream):
        dataset_embeddings.append(doc[embeddings_field])
        # dataset_titles.append(doc['title'])
        if i == dataset_size - 1:
            break
    dataset_embeddings = np.array(dataset_embeddings, dtype=numpy_types_dict[embedding_type])
    dataset_load_time = time.time() - start_time
    print(f"Loading {dataset_size} dataset embeddings took {dataset_load_time} seconds")

    # Continue iterating docs stream to get queries
    queries_embeddings = []
    # queries_titles = []
    start_time = time.time()
    for i, doc in enumerate(docs_stream):
        queries_embeddings.append(doc[embeddings_field])
        # dataset_titles.append(doc['title'])
        if i == queries_num - 1:
            break
    queries_embeddings = np.array(queries_embeddings, dtype=numpy_types_dict[embedding_type])
    queries_load_time = time.time() - start_time
    print(f"Loading {queries_num} queries embeddings took {queries_load_time} seconds")
    # return dataset_embeddings, dataset_titles, queries_embeddings, queries_titles
    return dataset_embeddings, queries_embeddings

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

def dataset_and_queries_SQ_embeddings(float_embeddings, float_queries_embeddings):
    start_time = time.time()
    dataset_sq_embeddings = quantize_embeddings(
        float_embeddings,
        precision="int8",
        calibration_embeddings=float_embeddings[:calibration_set_size],)
    dataset_time = time.time() - start_time
    print(f"Quantizing {len(dataset_sq_embeddings)} dataset embeddings took {dataset_time} seconds")

    start_time = time.time()
    query_sq_embeddings = quantize_embeddings(
        float_queries_embeddings,
        precision="int8",
        calibration_embeddings=float_embeddings[:calibration_set_size],)
    quries_time = time.time() - start_time
    print(f"Quantizing {len(query_sq_embeddings)} queries embeddings took {quries_time} seconds")
    return dataset_sq_embeddings, query_sq_embeddings

def batch_knn(queries_embeddings, dataset_embeddings, k = 10):
    res = []
    for query in queries_embeddings:
        res.append(knn_L2(query, dataset_embeddings, k))
    return res


def size_in_mb_pympler(obj):
    return asizeof.asizeof(obj) / (1024 * 1024)

def main():
    k = 10
    # Get GT ids
    # generate dataset embeddings using a dedicated model
    # dataset_and_queries_embeddings returns a dict of (type_embeddings, type_queries_embeddings) tuples
    # float32_vector_embeddings, float32_dataset_titles, float32_queries_embeddings, float32_queries_titles = dataset_and_queries_embeddings(float32_docs_stream, 'emb')
    float32_vector_embeddings, float32_queries_embeddings = dataset_and_queries_embeddings(float32_docs_stream, 'emb', "float")
    embed_size_mb_pympler = size_in_mb_pympler(float32_vector_embeddings)
    print(f"Memory of float_embeddings: {embed_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M float32 embeddings: {float(1_000_000/dataset_size) * (embed_size_mb_pympler/1024)} GB")

    gt_res = batch_knn(float32_queries_embeddings, float32_vector_embeddings, k = 10)
    int8_vector_embeddings, int8_queries_embeddings = dataset_and_queries_embeddings(int8_docs_stream, 'emb_int8', "int8")
    embed_size_mb_pympler = size_in_mb_pympler(int8_vector_embeddings)
    print(f"Memory of int8_embeddings: {embed_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M int8 embeddings: {float(1_000_000/dataset_size) * (embed_size_mb_pympler/1024)} GB")

    # int8_vector_embeddings, int8_dataset_titles, int8_queries_embeddings, int8_queries_titles = dataset_and_queries_embeddings(int8_docs_stream, 'emb_int8')

    # mismatch = 0
    # for i, title in enumerate(int8_dataset_titles):
    #     if title != float32_dataset_titles[i]:
    #         print(f"Title mismatch at index {i}. {title} != {float32_dataset_titles[i]}")
    #         mismatch += 1
    # print(f"Title mismatches: {mismatch}")

    print(f"Loaded int8 embeddings. Example vec slice = {int8_vector_embeddings[0][:10]}")
    correct = 0
    for i, query in enumerate(int8_queries_embeddings):
        res = knn_L2(query, int8_vector_embeddings, k = 10)
        correct += count_correct(gt_res[i], res)
    recall = correct / (k * queries_num)
    print(f"Embeddings from model\nRecall: ", recall)

    # generate dataset embeddings using a SQ
    sq_embeddings, sq_queries_embeddings = dataset_and_queries_SQ_embeddings(float32_vector_embeddings, float32_queries_embeddings)
    print(f"Quantized embeddings. Example vec slice = {sq_embeddings[0][:10]}")
    embed_size_mb_pympler = size_in_mb_pympler(sq_embeddings)

    print(f"Memory of sq_embeddings: {embed_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M SQ embeddings: {float(1_000_000/dataset_size) * (embed_size_mb_pympler/1024)} GB")

    correct = 0
    for i, query in enumerate(sq_queries_embeddings):
        res = knn_L2(query, sq_embeddings, k = 10)
        correct += count_correct(gt_res[i], res)
    recall = correct / (k * queries_num)
    print(f"Scalar quantization embeddings\nRecall: ", recall)


if __name__ == "__main__":
    main()
