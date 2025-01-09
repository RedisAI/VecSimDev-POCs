from ast import Dict
import numpy as np
from numpy.typing import NDArray
import time
import pickle

from torch import le

DATASET_SIZE = 1_000_000
CALIBRATION_SET_SIZE = DATASET_SIZE // 10
QUERIES_DATASET_SIZE = 100
QUERIES_NUM = 10
K = 10

numpy_types_dict = {
    "float32": np.float32,
    "int8": np.int8,
    "uint8": np.uint8
}
# quantize vectors pre-dimension
class ScalarQuantization:
    def __init__(self, dim, precision:str="uint8"):
        self.N = 255 # 2^B - 1
        self.dim = dim
        self.precision = precision
        if precision == "uint8":
            self.offset = 0
        elif precision == "int8":
            self.offset = 128

    def train(self, train_dataset: np.ndarray):
        # Assuming train_dataset is a numpy array with shape (n_train_vec, self.dim)
        self.x_min = train_dataset.min(axis=0)  # Find the minimum value in each dimension
        self.delta = (train_dataset.max(axis=0) - self.x_min) / self.N  # Calculate delta for each dimension

    def quantize(self, dataset: np.ndarray):
        q_vals = np.floor((dataset - self.x_min) / self.delta)
        # use int32 to avoid overflow if type is uint8
        q_vals = np.clip(q_vals, 0, self.N).astype(numpy_types_dict[self.precision])
        q_vals -= self.offset
        return q_vals

    def decompress(self, x):
        return (self.delta * (x + 0.5 + self.offset).astype(np.float32)) + self.x_min

    def get_quantization_params(self)-> Dict:
        return {"x_min": self.x_min, "delta": self.delta}

class Computer:
    def __init__(self, quantizer: ScalarQuantization):
        self.q = quantizer

    def IP_compressed_space(self, v1, v2):
        x_min = self.q.x_min
        delta = self.q.delta
        offset = self.q.offset
        # convert to a larger type to avoid overflow
        int_v1 = v1.astype(np.int32)
        int_v2 = v2.astype(np.int32)
        inner_product = (x_min ** 2) + x_min*delta*(int_v1 + int_v2 + 2*(offset + 0.5))  + (delta**2 )* (int_v1*int_v2 + (offset + 0.5)*(int_v1 + int_v2) + (offset + 0.5) ** 2)
        inner_product = inner_product.sum()
        # print(f"calculating on compressed space took {dur} seconds")
        return inner_product

    @staticmethod
    def IP_float32_domain(v1, v2):
        # Extract the diagonal elements which represent the inner products of corresponding pairs
        inner_products = np.dot(v1, v2)

        return inner_products

    def IP_decompressed_space(self, v1, v2):
        # print("\n decompresse and calculate IP")
        f_v1 = self.q.decompress(v1)
        f_v2 = self.q.decompress(v2)
        assert f_v1.dtype == np.float32, f"expected float32 but got {f_v1.dtype}"

        inner_product = self.IP_float32_domain(f_v1, f_v2)
        # print(f"calculating on decompressed space took {dur} seconds")
        return inner_product

    def timed_compute_batch(self, func, batch1, batch2, batch_size):
        res = []
        start = time.time()
        for i in range(batch_size):
            res.append(func(batch1[i], batch2[i]))
        dur = time.time() - start
        return np.array(res), dur

class EmbeddingLoader:

    def load_embeddings(self):
        embeddings_file = "float32_embeddings_1000000.pkl"
        print(f"Loading embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            self.dataset_embeddings = pickle.load(f)
        assert len(self.dataset_embeddings) == DATASET_SIZE, f"expected {DATASET_SIZE} but got {len(self.dataset_embeddings)}"
        assert self.dataset_embeddings.dtype == np.float32, f"expected float32 but got {self.dataset_embeddings.dtype}"

    def get_vectors(self, n_vectors):
        return self.dataset_embeddings[:n_vectors]

class Benchmark:
    def __init__(self, dataset, quantizer) -> None:
        assert len(dataset) % 2 == 0, f"dataset size must be even, but is {len(dataset)}"
        self.dataset = dataset
        self.q = quantizer
        self.comp = Computer(quantizer)

    @staticmethod
    def compute_error(gt_res, res):
        absolute_differences = np.abs(gt_res - res)

        # Calculate the average error
        average_error = np.mean(absolute_differences)

        return average_error

    def compare(self):
            # Split the dataset into two halves
        midpoint = len(self.dataset) // 2

        # Compute inner products
        gt_results, _ = self.comp.timed_compute_batch(self.comp.IP_float32_domain, self.dataset[:midpoint], self.dataset[midpoint:], midpoint)

        # quantize dataset
        q_dataset = self.q.quantize(self.dataset)

        # Compute inner products in compressed space
        comp_results, comp_time = self.comp.timed_compute_batch(self.comp.IP_compressed_space, q_dataset[:midpoint], q_dataset[midpoint:], midpoint)
        assert len(comp_results) == midpoint, f"expected {midpoint} but got {len(comp_results)}"
        # print("Inner product in compressed space : ", comp_results)
        print("\ncompressed space IP calc took ", comp_time, " seconds")
        # Calculate the error
        comp_results_error = self.compute_error(gt_results, comp_results)
        print("Error in compressed space: ", comp_results_error)

        #compute IP for decompressed space
        decomp_results, decomp_time  = self.comp.timed_compute_batch(self.comp.IP_decompressed_space, q_dataset[:midpoint], q_dataset[midpoint:], midpoint)
        # print("Inner product in decompressed space: ", decomp_results)
        print("\nDecompressed space IP calc took ", decomp_time, " seconds")
        decomp_results_error = self.compute_error(gt_results, decomp_results)
        print("Error in decompressed space: ", decomp_results_error)


def main():
    precision = "int8"
    print("Loading embeddings ...")
    embeddings_loader = EmbeddingLoader()
    embeddings_loader.load_embeddings()
    train_dataset = embeddings_loader.get_vectors(CALIBRATION_SET_SIZE)
    dataset =  embeddings_loader.get_vectors(DATASET_SIZE)
    print("train dataset shape = ", train_dataset.shape)
    print("dataset shape = ", dataset.shape)

    quantizer = ScalarQuantization(train_dataset.shape[1], precision)
    print("Creating quantizer for type = ", precision)

    print("\nTraining dataset ... ")
    start = time.time()
    quantizer.train(train_dataset)
    dur = time.time() - start
    print(f"Training took {dur} seconds. \nQuantization params = {quantizer.get_quantization_params()}")

    bm = Benchmark(dataset, quantizer)
    bm.compare()


if __name__ == "__main__":
    main()
