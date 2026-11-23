import os
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from implementation.client import Client
from implementation.topology import DynamicTopology
from implementation.aggregation import decentralized_fedavg

class ParallelDecentralizedEngine:
    """
    Parallel orchestration engine for Decentralized Federated Learning.
    Executes client local training, weight aggregation, and evaluation
    concurrently across worker threads.
    """
    def __init__(
        self,
        clients: Dict[int, Client],
        topology: DynamicTopology,
        max_workers: Optional[int] = None
    ):
        self.clients = clients
        self.num_clients = len(clients)
        self.topology = topology
        
        # Default workers based on CPU count or capped for balanced throughput
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(self.num_clients, max(4, cpu_count))
        else:
            self.max_workers = max_workers

    def parallel_train(self, epochs: int = 1) -> Dict[int, Tuple[float, float]]:
        """
        Executes local training for all clients in parallel.
        Returns:
            Dict[client_id, (train_loss, train_accuracy)]
        """
        results: Dict[int, Tuple[float, float]] = {}

        def _train_worker(cid: int, client: Client):
            loss, acc = client.train_epoch(epochs=epochs)
            return cid, (loss, acc)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_train_worker, cid, self.clients[cid])
                for cid in range(self.num_clients)
            ]
            for future in as_completed(futures):
                cid, res = future.result()
                results[cid] = res

        return results

    def parallel_exchange_and_aggregate(
        self,
        round_num: int
    ) -> Dict[int, List[int]]:
        """
        1. Obtains the dynamic neighbor graph for round_num.
        2. Gathers current weights from all clients in parallel.
        3. Aggregates neighbor weights using FedAvg in parallel.
        4. Updates each client's model weights.

        Returns:
            round_topology: Dict[client_id, List[neighbor_ids]]
        """
        # Step 1: Dynamic neighborhood generation for this iteration
        round_topology = self.topology.get_neighbors_for_round(round_num)

        # Step 2: Extract current model weights in parallel
        current_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        def _get_weights_worker(cid: int, client: Client):
            return cid, client.get_weights()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_get_weights_worker, cid, self.clients[cid])
                for cid in range(self.num_clients)
            ]
            for future in as_completed(futures):
                cid, w = future.result()
                current_weights[cid] = w

        # Step 3 & 4: Decentralized FedAvg aggregation and weight update in parallel
        def _aggregate_worker(cid: int, client: Client):
            neighbors = round_topology[cid]
            neighbor_weights = [current_weights[nid] for nid in neighbors]
            new_weights = decentralized_fedavg(
                local_weights=current_weights[cid],
                neighbor_weights_list=neighbor_weights
            )
            client.set_weights(new_weights)
            return cid

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_aggregate_worker, cid, self.clients[cid])
                for cid in range(self.num_clients)
            ]
            for future in as_completed(futures):
                future.result()

        return round_topology

    def parallel_evaluate(
        self,
        dataset_type: str = "val"
    ) -> Dict[int, Tuple[float, float]]:
        """
        Evaluates all clients on their local dataset ('val' or 'test') in parallel.
        Returns:
            Dict[client_id, (loss, accuracy)]
        """
        results: Dict[int, Tuple[float, float]] = {}

        def _eval_worker(cid: int, client: Client):
            loss, acc = client.evaluate(dataset_type=dataset_type)
            return cid, (loss, acc)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_eval_worker, cid, self.clients[cid])
                for cid in range(self.num_clients)
            ]
            for future in as_completed(futures):
                cid, res = future.result()
                results[cid] = res

        return results

    def run_round(
        self,
        round_num: int,
        local_epochs: int = 1
    ) -> Dict[str, any]:
        """
        Orchestrates one complete decentralized parallel iteration:
        1. Parallel local training
        2. Dynamic topology & parallel peer FedAvg aggregation
        3. Parallel local validation evaluation
        """
        t0 = time.time()
        
        # 1. Parallel Local Training
        train_results = self.parallel_train(epochs=local_epochs)
        
        # 2. Dynamic Peer Exchange & Aggregation
        round_topology = self.parallel_exchange_and_aggregate(round_num=round_num)
        
        # 3. Parallel Validation Evaluation
        val_results = self.parallel_evaluate(dataset_type="val")
        
        elapsed = time.time() - t0

        avg_train_loss = sum(r[0] for r in train_results.values()) / self.num_clients
        avg_train_acc = sum(r[1] for r in train_results.values()) / self.num_clients
        avg_val_loss = sum(r[0] for r in val_results.values()) / self.num_clients
        avg_val_acc = sum(r[1] for r in val_results.values()) / self.num_clients

        return {
            "round": round_num,
            "train_results": train_results,
            "val_results": val_results,
            "avg_train_loss": avg_train_loss,
            "avg_train_acc": avg_train_acc,
            "avg_val_loss": avg_val_loss,
            "avg_val_acc": avg_val_acc,
            "topology": round_topology,
            "round_duration": elapsed
        }
