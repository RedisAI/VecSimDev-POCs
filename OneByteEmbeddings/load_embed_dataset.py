from datasets import load_dataset
from sentence_transformers.quantization import quantize_embeddings
from scipy import spatial
import time
import numpy as np
import os
import pickle
from dotenv import load_dotenv
import cohere

# Load COHERE_API_KEY from .env file
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

# Get embeddings
# https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary
LANG = "en" #Use the English Wikipedia subset
int8_docs_stream = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", LANG, split="train", streaming=True)
float32_docs_stream = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", LANG, split="train", streaming=True)
queries_docs_stream = load_dataset("Cohere/miracl-en-queries-22-12", split="train", streaming=True)

DATASET_SIZE = 1_000_000
CALIBRATION_SET_SIZE = DATASET_SIZE // 10
QUERIES_NUM = 10
K = 10

numpy_types_dict = {
    "float32": np.float32,
    "int8": np.int8,
    "uint8": np.uint8
}

cohere_type_dict = {
    "float32": "float",
    "int8": "int8",
}

def load_field_from_stream(docs_stream, num_docs_to_load, field):
    res = []
    docs_stream = docs_stream.take(num_docs_to_load)
    for doc in docs_stream:
        res.append(doc[field])
    return res

def load_embeddings(docs_stream, embeddings_field, embedding_type):
    embeddings_file = f"{embedding_type}_embeddings_{DATASET_SIZE}.pkl"
    dataset_embeddings = []
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            dataset_embeddings = pickle.load(f)
    else:
        print("Embeddings file not found. Generating embeddings...")
        batch_size = CALIBRATION_SET_SIZE
        start_time = time.time()
        for i, processed_docs_num in enumerate(range(0, DATASET_SIZE, batch_size)):
            assert len(dataset_embeddings) == processed_docs_num, f"expected {len(dataset_embeddings)} == {processed_docs_num}"
            curr_docs_stream = docs_stream.skip(processed_docs_num)
            dataset_embeddings.extend(load_field_from_stream(curr_docs_stream, batch_size, embeddings_field))
            print(f"Done loading batch {i}, example slice: ", dataset_embeddings[-1][:5])
        dataset_embeddings = np.array(dataset_embeddings, dtype=numpy_types_dict[embedding_type])
        dataset_load_time = time.time() - start_time
        print(f"Loading {DATASET_SIZE} dataset embeddings took {dataset_load_time} seconds")

        with open(embeddings_file, 'wb') as f:
            pickle.dump(dataset_embeddings, f)
        print(f"Embeddings have been stored in {embeddings_file}")

    assert len(dataset_embeddings) == DATASET_SIZE
    return dataset_embeddings

def load_queries(queries_docs_stream, embedding_type):
    queries_embeddings_file = f"{embedding_type}_queries_{QUERIES_NUM}.pkl"
    queries_embeddings = []
    if os.path.exists(queries_embeddings_file):
        print(f"Loading queries from {queries_embeddings_file}")
        with open(queries_embeddings_file, 'rb') as f:
            queries_embeddings = pickle.load(f)
    else:
        print("Queries file not found. Generating embeddings...")
        queries_texts = []
        start_time = time.time()
        queries_texts.extend(load_field_from_stream(queries_docs_stream, QUERIES_NUM, "query"))
        queries_load_time = time.time() - start_time
        print(f"Loading {QUERIES_NUM} queries texts took {queries_load_time} seconds")
        # embed queries
        cohere_type = cohere_type_dict[embedding_type]
        queries_embeddings = co.embed(texts=queries_texts, model="embed-english-v3.0", input_type="search_query", embedding_types=[cohere_type]).embeddings
        queries_embeddings = getattr(queries_embeddings, cohere_type)
        with open(queries_embeddings_file, 'wb') as f:
            pickle.dump(queries_embeddings, f)
        print(f"Embeddings have been stored in {queries_embeddings_file}")

    queries_embeddings = np.array(queries_embeddings, dtype=numpy_types_dict[embedding_type])
    assert len(queries_embeddings) == QUERIES_NUM, f"expected {len(queries_embeddings)} == {QUERIES_NUM}"
    return queries_embeddings

def knn_L2(query, doc_embeddings, k=K):
    # compute the euclidean distance between the query and all the documents
    res = [(spatial.distance.euclidean(query, vec), id) for id, vec in enumerate(doc_embeddings)]
    res = sorted(res)
    # return the top k documents
    return res[0:k]

def knn_cosine(query, doc_embeddings, k=K):
    res = []
    # Compute the norm of the query vector
    query_norm = 1
    if query.dtype == np.int8:
        query = query.astype(np.int32)
        query_norm = np.linalg.norm(query)
    for id, vec in enumerate(doc_embeddings):
        # Convert document vector to numpy array with dtype float32
        # vec = np.array(vec, dtype=np.float32)
        # Compute the norm of the document vector
        vec_norm = 1
        if vec.dtype == np.int8:
            vec = vec.astype(np.int32)
            vec_norm = np.linalg.norm(vec)
        cosine_similarity = (np.dot(query, vec)) / (query_norm * vec_norm)
        # Compute cosine similarity and then convert to cosine distance
        cosine_distance = 1.0 - cosine_similarity
        res.append((cosine_distance, id))

    # Process the results as needed
    # Sort the results by distance
    res = sorted(res)
    # Return the top k documents
    return res[:k]

def batch_knn(queries_embeddings, dataset_embeddings, distance_func, k=K):
    res = []
    start_time = time.time()
    for query in queries_embeddings:
        res.append(distance_func(query, dataset_embeddings, k))
    batch_knn_time = time.time() - start_time
    print(f"knn {distance_func.__name__} for {len(queries_embeddings)} queries took {batch_knn_time} seconds")
    return res

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
        calibration_embeddings=float_embeddings[:CALIBRATION_SET_SIZE],)
    dataset_time = time.time() - start_time
    print(f"Quantizing {len(dataset_sq_embeddings)} dataset embeddings took {dataset_time} seconds")
    assert len(dataset_sq_embeddings) == DATASET_SIZE
    start_time = time.time()
    query_sq_embeddings = quantize_embeddings(
        float_queries_embeddings,
        precision="int8",
        calibration_embeddings=float_embeddings[:CALIBRATION_SET_SIZE],)
    quries_time = time.time() - start_time
    print(f"Quantizing {len(query_sq_embeddings)} queries embeddings took {quries_time} seconds")
    return dataset_sq_embeddings, query_sq_embeddings


def main():
    distance_func = knn_L2
    k = K
    print("Run BM with following parameters:"
          "\n\t DATASET_SIZE = ", DATASET_SIZE,
          "\n\t CALIBRATION_SET_SIZE = ", CALIBRATION_SET_SIZE,
          "\n\t QUERIES_NUM = ", QUERIES_NUM,
          "\n\t k = ", K,
          "\n\t distance_func = ", distance_func.__name__)

    print("Get float32 embeddings and queries")
    float32_vector_embeddings = load_embeddings(float32_docs_stream, 'emb', "float32")
    float32_queries_embeddings = load_queries(queries_docs_stream, "float32")
    print(f"\n\t Example vec slice = {float32_vector_embeddings[0][:5]}"
        f"\n\t Example query slice = {float32_queries_embeddings[0][:5]}\n")

    print("Get int8 embeddings and queries")
    int8_vector_embeddings = load_embeddings(int8_docs_stream, 'emb_int8', "int8")
    int8_queries_embeddings = load_queries(queries_docs_stream, "int8")
    print(f"\n\t Example vec slice = {int8_vector_embeddings[0][:5]}"
           f"\n\t Example query slice = {int8_queries_embeddings[0][:5]}\n")

    print(f"Calculate Ground truth (float32) IDs for {distance_func.__name__} search")
    gt_res = batch_knn(float32_queries_embeddings, float32_vector_embeddings, distance_func)
    print(f"float32 Example query_{QUERIES_NUM - 1} res: {gt_res[QUERIES_NUM - 1]}")

    print()
    print("calculate recall for int8 embeddings")
    start_time = time.time()
    correct = 0
    for i, query in enumerate(int8_queries_embeddings):
        res = distance_func(query, int8_vector_embeddings)
        correct += count_correct(gt_res[i], res)
    recall = correct / (k * QUERIES_NUM)
    recall_time = time.time() - start_time
    print(f"int8 Example query_{QUERIES_NUM - 1} res: {res}")
    print(f"int8 Embeddings from model search with k = {k} took {recall_time} seconds. \nRecall: {recall}\n")

    # generate dataset embeddings using a SQ
    print(f"Quantizing embeddings using calibration set of size {CALIBRATION_SET_SIZE}")
    sq_embeddings, sq_queries_embeddings = dataset_and_queries_SQ_embeddings(float32_vector_embeddings, float32_queries_embeddings)
    print(f"Quantized embeddings. Example vec slice = {sq_embeddings[0][:10]}")

    # print(f"Memory of sq_embeddings: {embed_size_mb_pympler} MB")
    # print(f"Estimated memory required for 1M SQ embeddings: {float(1_000_000/DATASET_SIZE) * (embed_size_mb_pympler/1024)} GB")

    start_time = time.time()
    correct = 0
    for i, query in enumerate(sq_queries_embeddings):
        res = distance_func(query, sq_embeddings)
        correct += count_correct(gt_res[i], res)
    recall = correct / (k * QUERIES_NUM)
    recall_time = time.time() - start_time
    print(f"SQ Example query_{QUERIES_NUM - 1} res: {res}")
    print(f"Scalar quantization embeddings search with k = {k} took {recall_time} seconds. \nRecall: ", recall)


if __name__ == "__main__":
    main()
