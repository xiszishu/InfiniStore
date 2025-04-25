import time
import random
import string
import asyncio
import argparse
import numpy as np
import uuid
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple
from datetime import datetime

import infinistore
import torch

class InfiniStoreClient:
    """Wrapper for InfiniStore async client interface using InfinityConnection"""
    
    def __init__(self, config, args):
        """Initialize with InfiniStore connection configuration"""
        self.conn = infinistore.InfinityConnection(config)
        self.name = "InfiniStore"
        self.block_size = args.block_size  # Default block size
        self.conn.connect()
    
    async def put(self, key, size, data_ptr):
        """Wrapper for rdma_write_cache_async
        
        Based on the actual method signature:
        async def rdma_write_cache_async(self, blocks: List[Tuple[str, int]], block_size: int, ptr: int)
        """
        try:
            if self.conn.rdma_connected:
            # Call the InfiniStore method
                await self.conn.rdma_write_cache_async([key], size, data_ptr)
            else:
                tcp_key = key[0]
                tcp_offset = key[1]
                await self.conn.tcp_write_cache(tcp_key, size, data_ptr)
            return True
        except Exception as e:
            print(f"Error in put operation: {e}")
            return False
    
    async def get(self, key, size, data_ptr):
        """Wrapper for rdma_read_cache_async
        
        Based on the actual method signature:
        async def rdma_read_cache_async(self, blocks: List[Tuple[str, int]], block_size: int, ptr: int)
        """
        try:
            # Call the InfiniStore method
            if self.conn.rdma_connected:
                result = await self.conn.rdma_read_cache_async([key], size, data_ptr)
            else:
                tcp_key = key[0]
                tcp_offset = key[1]
                result = self.conn.tcp_read_cache(tcp_key)
            
            # In a real implementation, you'd extract the data from the memory pointer
            # For this example, we'll return the raw result
            return result
        except Exception as e:
            print(f"Error in get operation: {e}")
            return None
    
    def register_mr(self, arr: torch.tensor):
        self.conn.register_mr(arr.data_ptr(), arr.numel() * arr.element_size())

    def __del__(self):
        self.conn.close()

class YCSBWorkload:
    """YCSB workload generator"""
    
    def __init__(self, client, args, record_count=1000, operation_count=1000, 
                 read_proportion=0.5, block_size = 1024, zipfian_constant=0.99):
        self.client = client
        self.record_count = record_count
        self.operation_count = operation_count
        self.read_proportion = read_proportion
        self.zipfian_constant = zipfian_constant
        self.keys = []
        self.block_size = block_size
        if args.rdma:
            src_device = "cpu" if args.src_gpu == -1 else "cuda:" + str(args.src_gpu)
            dst_device = "cpu" if args.dst_gpu == -1 else "cuda:" + str(args.dst_gpu)
        else:
            src_device = "cpu"
            dst_device = "cpu"
        self.write_buffer = torch.rand(
            self.record_count * block_size, device=src_device, dtype=torch.float32
        )
        self.value_size = block_size * self.write_buffer.element_size()
        self.read_buffer = torch.zeros(
            self.record_count * block_size, device=dst_device, dtype=torch.float32
        )
        self.generate_keys()
        if args.rdma:
            if src_device != "cpu":
                torch.cuda.synchronize(self.write_buffer.device)
            if dst_device != "cpu":
                torch.cuda.synchronize(self.read_buffer.device)
            self.client.register_mr(self.write_buffer)
            self.client.register_mr(self.read_buffer)

    def generate_keys(self):
        """Generate keys using YCSB's key pattern: 'user' + number"""
        keys = [str(uuid.uuid4()) for i in range(self.record_count)]
        offsets = [i * self.value_size for i in range(self.record_count)]
        self.keys = list(zip(keys, offsets))
    
    def zipfian_next(self):
        """Simple approximation of Zipfian distribution for key selection"""
        rank = int(random.random() ** self.zipfian_constant * self.record_count)
        return self.keys[min(rank, self.record_count - 1)]
    
    def next_operation(self):
        """Determine next operation (read or write) and key"""
        op_type = "get" if random.random() < self.read_proportion else "put"
        key = self.zipfian_next()
        size = self.value_size
        data_ptr = self.read_buffer.data_ptr() if op_type == "put" else self.write_buffer.data_ptr()
        
        return op_type, key, size, data_ptr

class YCSBBenchmark:
    """Benchmark runner for YCSB workloads on InfiniStore"""
    
    def __init__(self, client, workload):
        self.client = client
        self.workload = workload
        self.results = {
            "put": {"latencies": [], "throughput": 0},
            "get": {"latencies": [], "throughput": 0},
            "overall": {"latency": 0, "throughput": 0}
        }
    
    async def load_phase(self):
        """Load initial data into InfiniStore"""
        print(f"Loading {self.workload.record_count} records...")
        start_time = time.time()
        
        # Create a batch of tasks for parallel loading
        tasks = []
        for key in self.workload.keys:
            data_ptr = self.workload.write_buffer.data_ptr()
            tasks.append(self.client.put(key, self.workload.value_size, data_ptr))
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        throughput = self.workload.record_count / elapsed if elapsed > 0 else 0
        print(f"Load phase completed in {elapsed:.2f} seconds")
        print(f"Load throughput: {throughput:.2f} ops/sec")
        
        return throughput
    
    async def run_workload(self, concurrency=100):
        """Run the benchmark workload with specified concurrency"""
        print(f"Running workload with {self.workload.operation_count} operations...")
        print(f"Read proportion: {self.workload.read_proportion}")
        print(f"Concurrency: {concurrency}")
        
        # Create a semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        get_count = 0
        put_count = 0
        start_time = time.time()
        
        async def execute_operation(op_type, key, size, data_ptr):
            nonlocal get_count, put_count
            
            async with sem:
                op_start = time.time()
                if op_type == "get":
                    await self.client.get(key, size, data_ptr)
                    get_count += 1
                else:
                    await self.client.put(key, size, data_ptr)
                    put_count += 1
                op_end = time.time()
                
                latency = (op_end - op_start) * 1000  # ms
                self.results[op_type]["latencies"].append(latency)
        
        # Generate all operations first
        operations = [self.workload.next_operation() for _ in range(self.workload.operation_count)]

        # block_size = args.block_size * 1024 // 4  # element size is 4 bytes
        # num_of_blocks = args.size * 1024 * 1024 // (args.block_size * 1024)
        
        # Execute all operations with controlled concurrency
        tasks = []
        for op_type, key, size, data_ptr in operations:
            tasks.append(execute_operation(op_type, key, size, data_ptr))
        
        # Wait for all operations to complete
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        total_ops = get_count + put_count
        overall_throughput = total_ops / elapsed if elapsed > 0 else 0
        
        # Get throughput
        get_throughput = get_count / elapsed if elapsed > 0 else 0
        self.results["get"]["throughput"] = get_throughput
        
        # Put throughput
        put_throughput = put_count / elapsed if elapsed > 0 else 0
        self.results["put"]["throughput"] = put_throughput
        
        # Overall metrics
        self.results["overall"]["throughput"] = overall_throughput
        self.results["overall"]["latency"] = elapsed * 1000 / total_ops if total_ops > 0 else 0
        
        return self.results
    
    def print_results(self):
        """Print benchmark results"""
        print("\n===== BENCHMARK RESULTS =====")
        print(f"Store: {self.client.name}")
        print(f"Record count: {self.workload.record_count}")
        print(f"Operation count: {self.workload.operation_count}")
        print(f"Read proportion: {self.workload.read_proportion}")
        print(f"Value size: {self.workload.value_size} bytes")
        
        print("\n--- THROUGHPUT ---")
        print(f"Overall throughput: {self.results['overall']['throughput']:.2f} ops/sec")
        print(f"GET throughput: {self.results['get']['throughput']:.2f} ops/sec")
        print(f"PUT throughput: {self.results['put']['throughput']:.2f} ops/sec")
        
        print("\n--- LATENCY (ms) ---")
        if self.results["get"]["latencies"]:
            print(f"GET Avg: {np.mean(self.results['get']['latencies']):.2f} ms")
            print(f"GET Min: {min(self.results['get']['latencies']):.2f} ms")
            print(f"GET Max: {max(self.results['get']['latencies']):.2f} ms")
            print(f"GET 95th: {np.percentile(self.results['get']['latencies'], 95):.2f} ms")
            print(f"GET 99th: {np.percentile(self.results['get']['latencies'], 99):.2f} ms")
        
        if self.results["put"]["latencies"]:
            print(f"PUT Avg: {np.mean(self.results['put']['latencies']):.2f} ms")
            print(f"PUT Min: {min(self.results['put']['latencies']):.2f} ms")
            print(f"PUT Max: {max(self.results['put']['latencies']):.2f} ms")
            print(f"PUT 95th: {np.percentile(self.results['put']['latencies'], 95):.2f} ms")
            print(f"PUT 99th: {np.percentile(self.results['put']['latencies'], 99):.2f} ms")
        
        print(f"Overall Avg: {self.results['overall']['latency']:.2f} ms")
    
    def plot_latency_distribution(self):
        """Plot the latency distribution"""
        plt.figure(figsize=(12, 6))
        
        # Only plot if we have data
        if self.results["get"]["latencies"]:
            plt.subplot(1, 2, 1)
            plt.hist(self.results["get"]["latencies"], bins=20, alpha=0.7, label='GET')
            plt.title('GET Latency Distribution')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Count')
        
        if self.results["put"]["latencies"]:
            plt.subplot(1, 2, 2)
            plt.hist(self.results["put"]["latencies"], bins=20, alpha=0.7, label='PUT')
            plt.title('PUT Latency Distribution')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Count')
        
        plt.tight_layout()
        file_name = f"infinistore_throughput_vs_concurrency_{timestamp}.png"
        plt.savefig(file_name)
        print(f"Throughput vs concurrency plot saved to {file_name}")
    
    def plot_throughput_vs_concurrency(self, concurrency_levels, results_list):
        """Plot throughput vs concurrency levels"""
        plt.figure(figsize=(10, 6))
        
        # Extract throughput values from results
        throughputs = [res["overall"]["throughput"] for res in results_list]
        
        plt.plot(concurrency_levels, throughputs, marker='o', linestyle='-')
        plt.title('Throughput vs Concurrency')
        plt.xlabel('Concurrency Level')
        plt.ylabel('Throughput (ops/sec)')
        plt.grid(True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"infinistore_throughput_vs_concurrency_{timestamp}.png"
        plt.savefig(file_name)
        print(f"Throughput vs concurrency plot saved to {file_name}")

async def run_benchmark(args):
    """Run the benchmark with given arguments"""
    # Create InfiniStore configuration
    config = infinistore.ClientConfig(
        host_addr=args.server,
        service_port=args.service_port,
        dev_name=args.dev_name,
        ib_port=args.ib_port,
        link_type=args.link_type,
        log_level="warning",
    )

    if args.rdma:
        config.connection_type = infinistore.TYPE_RDMA
    else:
        config.connection_type = infinistore.TYPE_TCP
    
    # Create InfiniStore client
    client = InfiniStoreClient(config, args)
    num_of_blocks = args.size * 1024 * 1024 // (args.block_size * 1024)
    
    # Create workload
    workload = YCSBWorkload(
        client=client,
        args=args,
        record_count=args.records,
        operation_count=args.operations,
        read_proportion=args.read_proportion,
        block_size=args.block_size
    )
    
    # Create benchmark
    benchmark = YCSBBenchmark(client, workload)
    
    # Run load phase
    await benchmark.load_phase()
    
    if args.concurrency_test:
        # Run with different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50, 100, 200, 500]
        all_results = []
        
        for concurrency in concurrency_levels:
            print(f"\nRunning with concurrency level: {concurrency}")
            # Reset results for this run
            benchmark.results = {
                "put": {"latencies": [], "throughput": 0},
                "get": {"latencies": [], "throughput": 0},
                "overall": {"latency": 0, "throughput": 0}
            }
            
            results = await benchmark.run_workload(concurrency=concurrency)
            all_results.append(results.copy())  # Store a copy of the results
        
        # Plot throughput vs concurrency
        benchmark.plot_throughput_vs_concurrency(concurrency_levels, all_results)
    else:
        # Run with single concurrency level
        await benchmark.run_workload(concurrency=args.concurrency)
        benchmark.print_results()
        benchmark.plot_latency_distribution()

def main():
    parser = argparse.ArgumentParser(description='YCSB Benchmark for InfiniStore')
    parser.add_argument('--hosts', default='localhost', 
                        help='Comma-separated list of InfiniStore hosts')
    parser.add_argument('--port', type=int, default=31000,
                        help='InfiniStore port number')
    parser.add_argument('--records', type=int, default=1000,
                        help='Number of records to load initially')
    parser.add_argument('--operations', type=int, default=10000,
                        help='Number of operations to perform')
    parser.add_argument('--read-proportion', type=float, default=0.5,
                        help='Proportion of read operations (0.0-1.0)')
    parser.add_argument('--value-size', type=int, default=100,
                        help='Size of values in bytes')
    parser.add_argument('--block-size', type=int, default=4096,
                        help='Block size in bytes')
    parser.add_argument('--concurrency', type=int, default=100,
                        help='Number of concurrent operations')
    parser.add_argument('--concurrency-test', action='store_true',
                        help='Run tests with different concurrency levels')
    parser.add_argument(
        "--rdma",
        required=False,
        action="store_true",
        help="use rdma connection, default False",
        default=False,
    )

    parser.add_argument(
        "--server",
        required=False,
        help="connect to which server, default 127.0.0.1",
        default="127.0.0.1",
        type=str,
    )
    parser.add_argument(
        "--service-port",
        required=False,
        type=int,
        default=22345,
        help="port for data plane, default 22345",
    )
    parser.add_argument(
        "--dev-name",
        required=False,
        default="mlx5_1",
        help="Use IB device <dev> (default first device found)",
        type=str,
    )
    parser.add_argument(
        "--size",
        required=False,
        type=int,
        default=128,
        help="size for benchmarking, unit: MB, default 128",
    )
    parser.add_argument(
        "--src-gpu",
        required=False,
        type=int,
        default=-1,
        help="gpu# for data write from, default 0",
    )
    parser.add_argument(
        "--dst-gpu",
        required=False,
        type=int,
        default=-1,
        help="gpu# for data read to, default 1",
    )
    parser.add_argument(
        "--ib-port",
        required=False,
        type=int,
        default=1,
        help="use port <port> of IB device (default 1)",
    )
    parser.add_argument(
        "--link-type",
        required=False,
        default="IB",
        help="IB or Ethernet, default IB",
        type=str,
    )

    args = parser.parse_args()
    
    # Run the benchmark
    asyncio.run(run_benchmark(args))

if __name__ == "__main__":
    main()