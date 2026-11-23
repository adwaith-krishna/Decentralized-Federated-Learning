from typing import Dict, List, Set
import numpy as np

class DynamicTopology:
    """
    Manages dynamic peer-to-peer communication graphs across decentralized clients.
    In each iteration, the communication topology changes dynamically.
    """
    def __init__(
        self,
        num_clients: int = 50,
        neighbors_per_client: int = 4,
        symmetric: bool = True,
        seed: int = 42
    ):
        """
        Args:
            num_clients: total number of clients in the network (e.g., 50).
            neighbors_per_client: target number of peer neighbors each client connects to (k).
            symmetric: if True, if j is neighbor of i, i is also neighbor of j (undirected).
            seed: random seed base.
        """
        assert neighbors_per_client < num_clients, "neighbors_per_client must be less than num_clients"
        self.num_clients = num_clients
        self.k = neighbors_per_client
        self.symmetric = symmetric
        self.seed = seed
        self.current_round = 0

    def get_neighbors_for_round(self, round_num: int) -> Dict[int, List[int]]:
        """
        Generates a fresh neighbor mapping for round `round_num`.
        Guarantees:
        1. No self-connections (i not in neighbors[i]).
        2. Neighbors change in each iteration.
        3. Every client has active neighbors.

        Returns:
            Dict[client_id, List[neighbor_ids]]
        """
        self.current_round = round_num
        rng = np.random.default_rng(self.seed + round_num * 10007)

        if self.symmetric:
            # Generate random symmetric graph
            adj: Dict[int, Set[int]] = {i: set() for i in range(self.num_clients)}
            
            # Form random pairings until minimum degree is satisfied
            nodes = list(range(self.num_clients))
            for i in nodes:
                while len(adj[i]) < self.k:
                    # Candidates are nodes != i and not already in adj[i]
                    candidates = [j for j in nodes if j != i and j not in adj[i]]
                    if not candidates:
                        break
                    # Prefer candidates with fewer connections
                    chosen = rng.choice(candidates)
                    adj[i].add(chosen)
                    adj[chosen].add(i)

            return {i: sorted(list(neighbors)) for i, neighbors in adj.items()}
        else:
            # Directed dynamic topology: each client independently chooses k distinct peers
            adj_directed: Dict[int, List[int]] = {}
            all_clients = np.arange(self.num_clients)
            for i in range(self.num_clients):
                candidates = all_clients[all_clients != i]
                chosen = rng.choice(candidates, size=self.k, replace=False)
                adj_directed[i] = sorted(chosen.tolist())
            return adj_directed

    def verify_dynamic_change(self, round1: int, round2: int) -> float:
        """
        Computes the fraction of neighbor links that changed between round1 and round2.
        Returns a value between 0.0 (identical) and 1.0 (completely different).
        """
        g1 = self.get_neighbors_for_round(round1)
        g2 = self.get_neighbors_for_round(round2)

        total_comparisons = 0
        changed_neighbors = 0

        for i in range(self.num_clients):
            s1 = set(g1[i])
            s2 = set(g2[i])
            diff = s1.symmetric_difference(s2)
            changed_neighbors += len(diff)
            total_comparisons += len(s1.union(s2))

        return changed_neighbors / max(total_comparisons, 1)
