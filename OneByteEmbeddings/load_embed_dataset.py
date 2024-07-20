from tracemalloc import start
from datasets import load_dataset
import cohere
from sentence_transformers.quantization import quantize_embeddings
from scipy import spatial
import time
from pympler import asizeof
import numpy as np
import os
import pickle

# Get embeddings
# https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary
lang = "simple" #Use the Simple English Wikipedia subset
int8_docs_stream = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", lang, split="train", streaming=True)
float32_docs_stream = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split="train", streaming=True)

# dataset_size = 100_000_000
dataset_size = 100
calibration_set_size = int(dataset_size/10)
queries_num = 10

numpy_types_dict = {
    "float32": np.float32,
    "int8": np.int8,
    "uint8": np.uint8
}

def dataset_and_queries_embeddings(docs_stream, embeddings_field, embedding_type):
    embeddings_file = f"{embedding_type}_embeddings_{dataset_size}_vec_{queries_num}_queries.pkl"
    embeddings = {}
    # Check if embeddings file exists
    if os.path.exists(embeddings_file):
        print(f"Loading vec and queries embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
            # Verify the expected structure and content of the loaded embeddings
            if not isinstance(embeddings, dict) or 'dataset_embeddings' not in embeddings or 'queries_embeddings' not in embeddings:
                raise ValueError("Loaded embeddings do not have the expected format or keys.")
        dataset_embeddings = embeddings['dataset_embeddings']
        queries_embeddings = embeddings['queries_embeddings']
        # Additional checks to ensure data integrity and type
        if not isinstance(dataset_embeddings, np.ndarray) or not isinstance(queries_embeddings, np.ndarray):
            raise TypeError("Loaded embeddings are not numpy arrays as expected.")

    else:
        print("Embeddings file not found. Generating embeddings...")
        dataset_embeddings = []
        # dataset_titles = []
        start_time = time.time()
        for i, doc in enumerate(docs_stream):
            dataset_embeddings.append(doc[embeddings_field])
            # dataset_titles.append(doc['title'])
            if i == dataset_size - 1:
                break
        assert len(dataset_embeddings) == dataset_size
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
        assert len(queries_embeddings) == queries_num
        queries_embeddings = np.array(queries_embeddings, dtype=numpy_types_dict[embedding_type])
        queries_load_time = time.time() - start_time
        print(f"Loading {queries_num} queries embeddings took {queries_load_time} seconds")

        # Store both embeddings in a dictionary
        embeddings['dataset_embeddings'] = dataset_embeddings
        embeddings['queries_embeddings'] = queries_embeddings

        # Save embeddings to file
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings have been stored in {embeddings_file}")

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
    assert len(dataset_sq_embeddings) == dataset_size
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
    start_time = time.time()
    for query in queries_embeddings:
        res.append(knn_L2(query, dataset_embeddings, k))
    batch_knn_time = time.time() - start_time
    print(f"knn L2 for {len(queries_embeddings)} queries took {batch_knn_time} seconds")
    return res


def size_in_mb_pympler(obj):
    return asizeof.asizeof(obj) / (1024 * 1024)

def main():
    k = 10

    # float32_vector_embeddings, float32_dataset_titles, float32_queries_embeddings, float32_queries_titles = dataset_and_queries_embeddings(float32_docs_stream, 'emb')
    float32_vector_embeddings, float32_queries_embeddings = dataset_and_queries_embeddings(float32_docs_stream, 'emb', "float32")
    print("type of float32_vector_embeddings: ", type(float32_vector_embeddings))
    print(f"Loaded float32 embeddings and queries. \n\t Example vec slice = {float32_vector_embeddings[0][:5]}"
        f"\n\t Example query slice = {float32_queries_embeddings[0][:5]}")
    embed_size_mb_pympler = size_in_mb_pympler(float32_vector_embeddings)
    print(f"Memory of float_embeddings: {embed_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M float32 embeddings: {float(1_000_000/dataset_size) * (embed_size_mb_pympler/1024)} GB")

    int8_vector_embeddings, int8_queries_embeddings = dataset_and_queries_embeddings(int8_docs_stream, 'emb_int8', "int8")
    print(f"Loaded int8 embeddings and queries. \n\t Example vec slice = {int8_vector_embeddings[0][:5]}"
           f"\n\t Example query slice = {int8_queries_embeddings[0][:5]}")
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


    gt_res = batch_knn(float32_queries_embeddings, float32_vector_embeddings, k = 10)

    start_time = time.time()
    correct = 0
    for i, query in enumerate(int8_queries_embeddings):
        res = knn_L2(query, int8_vector_embeddings, k = 10)
        correct += count_correct(gt_res[i], res)
    recall = correct / (k * queries_num)
    recall_time = time.time() - start_time
    print(f"Embeddings from model search with k = {k} took {recall_time} seconds. \nRecall: ", recall)

    # generate dataset embeddings using a SQ
    print(f"Quantizing embeddings using calibration set of size {calibration_set_size}")
    sq_embeddings, sq_queries_embeddings = dataset_and_queries_SQ_embeddings(float32_vector_embeddings, float32_queries_embeddings)
    print(f"Quantized embeddings. Example vec slice = {sq_embeddings[0][:10]}")
    embed_size_mb_pympler = size_in_mb_pympler(sq_embeddings)

    print(f"Memory of sq_embeddings: {embed_size_mb_pympler} MB")
    print(f"Estimated memory required for 1M SQ embeddings: {float(1_000_000/dataset_size) * (embed_size_mb_pympler/1024)} GB")

    start_time = time.time()
    correct = 0
    for i, query in enumerate(sq_queries_embeddings):
        res = knn_L2(query, sq_embeddings, k = 10)
        correct += count_correct(gt_res[i], res)
    recall = correct / (k * queries_num)
    recall_time = time.time() - start_time
    print(f"Scalar quantization embeddings search with k = {k} took {recall_time} seconds. \nRecall: ", recall)


if __name__ == "__main__":
    main()
