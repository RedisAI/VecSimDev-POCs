from datasets import load_dataset
from sentence_transformers.quantization import quantize_embeddings
from scipy import spatial
import time
import numpy as np
import os
import pickle
from dotenv import load_dotenv
import cohere
import csv

# Load COHERE_API_KEY from .env file
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

# Constants
LANG = "en" # Use the English Wikipedia subset
DATASET_SIZE = 1_000_000
CALIBRATION_SET_SIZE = DATASET_SIZE // 1
QUERIES_DATASET_SIZE = 100
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

dataset_embed_type_dict = {
    "float32": "emb",
    "int8": "emb_int8"
}

VECTOR_TEXT = 0
QUERY_TEXT = 1

class TextsCache:
    def __init__(self):
        self.cache_file = 'texts_cache.pkl'
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache_list = pickle.load(f)
        else:
            self.cache_list = [{}, {}]  # [vecs_text_cache, queries_text_cache]

    def get(self, text_type, id):
        return self.cache_list[text_type].get(id, None)

    def set(self, text_type, id, value):
        self.cache_list[text_type][id] = value
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache_list, f)


class EmbeddingLoader:
    def __init__(self, dataset_stream, queries_stream, queries_model, embedding_type, cache):
        self.dataset_stream = dataset_stream
        self.queries_stream = queries_stream
        self.embedding_type = embedding_type
        self.queries_model = queries_model
        self.cache = cache

    @staticmethod
    def load_field_from_stream(docs_stream, field, num_docs_to_load, offset=0):
        docs_stream = docs_stream.skip(offset)
        docs_stream = docs_stream.take(num_docs_to_load)
        res = []
        for doc in docs_stream:
            res.append(doc[field])
        return res

    def load_embeddings(self):
        embeddings_file = f"{self.embedding_type}_embeddings_{DATASET_SIZE}.pkl"
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
                dataset_embeddings.extend(self.load_field_from_stream(self.dataset_stream, dataset_embed_type_dict[self.embedding_type], batch_size, processed_docs_num))
                print(f"Done loading batch {i}, example slice: ", dataset_embeddings[-1][:5])
            dataset_embeddings = np.array(dataset_embeddings, dtype=numpy_types_dict[self.embedding_type])
            dataset_load_time = time.time() - start_time
            print(f"Loading {DATASET_SIZE} dataset embeddings took {dataset_load_time} seconds")

            with open(embeddings_file, 'wb') as f:
                pickle.dump(dataset_embeddings, f)
            print(f"Embeddings have been stored in {embeddings_file}")

        assert len(dataset_embeddings) == DATASET_SIZE
        return dataset_embeddings

    def load_queries(self):
        queries_embeddings_file = f"{self.embedding_type}_queries_{QUERIES_DATASET_SIZE}_{self.queries_model}.pkl"
        queries_embeddings = []
        if os.path.exists(queries_embeddings_file):
            print(f"Loading queries from {queries_embeddings_file}")
            with open(queries_embeddings_file, 'rb') as f:
                queries_embeddings = pickle.load(f)
        else:
            print("Queries file not found. Generating embeddings...")
            queries_texts = self.load_field_from_stream(self.queries_stream, "query", QUERIES_DATASET_SIZE)
            for i, query_text in enumerate(queries_texts):
                self.cache.set(QUERY_TEXT, i, query_text)
            start_time = time.time()
            queries_embeddings = co.embed(
                texts=queries_texts,
                model=self.queries_model,
                input_type="search_query",
                embedding_types=[cohere_type_dict[self.embedding_type]]
            ).embeddings
            queries_embeddings = getattr(queries_embeddings, cohere_type_dict[self.embedding_type])
            queries_load_time = time.time() - start_time
            print(f"Loading {QUERIES_DATASET_SIZE} queries texts took {queries_load_time} seconds")

            with open(queries_embeddings_file, 'wb') as f:
                pickle.dump(queries_embeddings, f)
            print(f"Embeddings have been stored in {queries_embeddings_file}")

        queries_embeddings = np.array(queries_embeddings, dtype=numpy_types_dict[self.embedding_type])
        assert len(queries_embeddings) == QUERIES_DATASET_SIZE
        return queries_embeddings

    def get_vec_text_at_idx(self, idx):
        text = self.cache.get(VECTOR_TEXT, idx)
        if text is None:
            print(f"Cacheing vec {idx}")
            text = self.load_field_from_stream(self.dataset_stream, "text", 1, idx)[0]
            self.cache.set(VECTOR_TEXT, idx, text)
        else:
            print(f"Cache hit for vec at index {idx}")
        return text

    def get_query_text_at_idx(self, idx):
        text = self.cache.get(QUERY_TEXT, idx)
        if text is None:
            print(f"Cacheing query {idx}")
            text = self.load_field_from_stream(self.queries_stream, "query", 1, idx)[0]
            self.cache.set(QUERY_TEXT, idx, text)
        else:
            print(f"Cache hit for query at index {idx}")
        return text


RES_ID=1
class DistanceCalculator:
    @staticmethod
    def knn_L2(query, doc_embeddings, k=K):
        res = [(spatial.distance.euclidean(query, vec), id) for id, vec in enumerate(doc_embeddings)]
        res = sorted(res)
        return res[:k]

    @staticmethod
    def knn_cosine(query, doc_embeddings, k=K):
        res = []
        if query.dtype == np.int8:
            query = query.astype(np.int32)
        query_norm = np.linalg.norm(query)
        for id, vec in enumerate(doc_embeddings):
            if vec.dtype == np.int8:
                vec = vec.astype(np.int32)
            vec_norm = np.linalg.norm(vec)
            cosine_similarity = np.dot(query, vec) / (query_norm * vec_norm)
            cosine_distance = 1.0 - cosine_similarity
            res.append((cosine_distance, id))
        res = sorted(res)
        return res[:k]

class QuantizationProcessor:
    @staticmethod
    def dataset_and_queries_SQ_embeddings(float_embeddings, float_queries_embeddings):
        start_time = time.time()
        dataset_sq_embeddings = quantize_embeddings(
            float_embeddings,
            precision="int8",
            calibration_embeddings=float_embeddings[:CALIBRATION_SET_SIZE],
        )
        dataset_time = time.time() - start_time
        print(f"Quantizing {len(dataset_sq_embeddings)} dataset embeddings took {dataset_time} seconds")

        start_time = time.time()
        query_sq_embeddings = quantize_embeddings(
            float_queries_embeddings,
            precision="int8",
            calibration_embeddings=float_embeddings[:CALIBRATION_SET_SIZE],
        )
        queries_time = time.time() - start_time
        print(f"Quantizing {len(query_sq_embeddings)} queries embeddings took {queries_time} seconds")
        return dataset_sq_embeddings, query_sq_embeddings

class Benchmark:
    def __init__(self, float32_loader, int8_loader, results_file, k=K):
        self.float32_loader = float32_loader
        self.int8_loader = int8_loader
        self.k = k
        self.results_file = results_file

    def write_recall_to_csv(self, func_name, recall_int8, recall_SQ):
        csv_file_path = self.results_file
        # Check if the CSV file exists
        if not os.path.exists(csv_file_path):
            # Create a new file and write the header
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['metric', 'int8', 'SQ'])

        # Append the recall values
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([func_name, recall_int8, recall_SQ])

    def count_correct(self, gt_results, results):
        correct = 0
        for res in results:
            for gt_res in gt_results:
                if res[1] == gt_res[1]:
                    correct += 1
                    break
        return correct

    def batch_knn(self, queries_num, queries_embeddings, dataset_embeddings, distance_func, k=K):
        res = []
        start_time = time.time()
        for query in queries_embeddings[:queries_num]:
            res.append(distance_func(query, dataset_embeddings, k))
        batch_knn_time = time.time() - start_time
        print(f"knn {distance_func.__name__} for {queries_num} queries took {batch_knn_time} seconds")
        assert len(res) == queries_num, f"expected {len(queries_embeddings)} == {len(res)} == {queries_num}"
        return res

    def compute_recall(self, distance_func):
        print(f"Running benchmark with distance function: {distance_func.__name__}")

        float32_vector_embeddings = self.float32_loader.load_embeddings()
        float32_queries_embeddings = self.float32_loader.load_queries()
        print(f"\n\t Example vec slice = {float32_vector_embeddings[0][:5]}"
              f"\n\t Example query slice = {float32_queries_embeddings[0][:5]}\n")

        int8_vector_embeddings = self.int8_loader.load_embeddings()
        int8_queries_embeddings = self.int8_loader.load_queries()
        print(f"\n\t Example vec slice = {int8_vector_embeddings[0][:5]}"
              f"\n\t Example query slice = {int8_queries_embeddings[0][:5]}\n")

        print(f"Calculate Ground truth (float32) IDs for {distance_func.__name__} search with {QUERIES_NUM} queries")
        gt_res = self.batch_knn(QUERIES_NUM, float32_queries_embeddings, float32_vector_embeddings, distance_func)
        print(f"float32 Example query_{QUERIES_NUM - 1} res: {gt_res[QUERIES_NUM - 1]}")
        print(f"Best result for query_{QUERIES_NUM - 1}:")
        print(f"{self.print_query_answer(QUERIES_NUM - 1, gt_res[QUERIES_NUM - 1][0][RES_ID], self.float32_loader)}")

        print()
        print("Calculate recall for int8 embeddings")
        start_time = time.time()
        correct = 0
        for i, query in enumerate(int8_queries_embeddings[:QUERIES_NUM]):
            res = distance_func(query, int8_vector_embeddings)
            correct += self.count_correct(gt_res[i], res)
        int8_recall = correct / (K * QUERIES_NUM)
        recall_time = time.time() - start_time
        print(f"int8 Embeddings from model search with k = {K} took {recall_time} seconds. \nRecall: {int8_recall}\n")
        print(f"int8 Example query_{QUERIES_NUM - 1} res: {res}")
        print(f"Best result for query_{QUERIES_NUM - 1}:")
        print(f"{self.print_query_answer(QUERIES_NUM - 1, res[0][RES_ID], self.int8_loader)}")


        print(f"Quantizing embeddings using calibration set of size {CALIBRATION_SET_SIZE}")
        sq_embeddings, sq_queries_embeddings = QuantizationProcessor.dataset_and_queries_SQ_embeddings(float32_vector_embeddings, float32_queries_embeddings)
        print(f"Quantized embeddings. Example vec slice = {sq_embeddings[0][:10]}")

        start_time = time.time()
        correct = 0
        for i, query in enumerate(sq_queries_embeddings[:QUERIES_NUM]):
            res = distance_func(query, sq_embeddings)
            correct += self.count_correct(gt_res[i], res)
        SQ_recall = correct / (K * QUERIES_NUM)
        recall_time = time.time() - start_time
        print(f"Scalar quantization embeddings search with k = {K} took {recall_time} seconds. \nRecall: {SQ_recall}")
        print(f"SQ Example query_{QUERIES_NUM - 1} res: {res}")
        print(f"Best result for query_{QUERIES_NUM - 1}:")
        self.print_query_answer(QUERIES_NUM - 1, res[0][RES_ID], self.int8_loader)

        self.write_recall_to_csv(distance_func.__name__, int8_recall, SQ_recall)

    @staticmethod
    def print_query_answer(query_idx: int, vec_idx: int, loader):
        print()
        print(f"\nQuestion: {loader.get_query_text_at_idx(query_idx)}")
        answer_text = loader.get_vec_text_at_idx(vec_idx)
        print(f"\nAnswer: {answer_text}")



def main():
    print("Run BM with following parameters:"
        "\n\t DATASET_SIZE = ", DATASET_SIZE,
        "\n\t CALIBRATION_SET_SIZE = ", CALIBRATION_SET_SIZE,
        "\n\t QUERIES_DATASET_SIZE = ", QUERIES_DATASET_SIZE,
        "\n\t QUERIES_NUM = ", QUERIES_NUM,
        "\n\t k = ", K)

    texts_cache = TextsCache()
    float32_loader = EmbeddingLoader(
        load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", LANG, split="train", streaming=True),
        load_dataset("Cohere/miracl-en-queries-22-12", split="train", streaming=True),
        "embed-multilingual-v3.0",
        "float32",
        texts_cache
    )
    int8_loader = EmbeddingLoader(
        load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", LANG, split="train", streaming=True),
        load_dataset("Cohere/miracl-en-queries-22-12", split="train", streaming=True),
        "embed-multilingual-v3.0",
        "int8",
        texts_cache
    )

    csv_file_name = f"{DATASET_SIZE}_vecs_{CALIBRATION_SET_SIZE}_calibration_{QUERIES_DATASET_SIZE}_queries_{QUERIES_NUM}_k_{K}_recall.csv"
    benchmark = Benchmark(float32_loader, int8_loader, csv_file_name)
    benchmark.compute_recall(DistanceCalculator.knn_cosine)
    benchmark.compute_recall(DistanceCalculator.knn_L2)

if __name__ == "__main__":
    main()
