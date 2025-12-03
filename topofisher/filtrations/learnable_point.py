"""
Learnable filtration for point clouds using flag complex.

This module implements a learnable filtration that transforms point clouds
using a neural network to compute vertex filtration values. The network
takes k-nearest neighbor distances as input and outputs a filtration value
for each vertex.

Pipeline:
    Point Cloud → kNN distances → MLP → Vertex Filtration
                                              ↓
                    Flag Complex (alpha/rips) with learned filtration
                                              ↓
                              Persistence Diagrams (differentiable)

The key insight is that while persistence pairing is computed via GUDHI
(non-differentiable), the actual birth/death values are extracted via
torch.gather (differentiable), enabling gradient flow through the MLP.

Based on TensorFlow implementation in:
    external/topofisher_tensorflow/topofisher/filtrations/tensorflow/flag_layer.py
    external/topofisher_tensorflow/topofisher/filtrations/tensorflow/dtm_layer.py

TODO: Clean up and refactor this code:
    - Consolidate extract_diagrams_single logic
    - Add batched kNN computation for efficiency
    - Better error handling for edge cases
    - Add support for higher homology dimensions (H2+)
"""
from typing import List, Optional, Literal
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm


class VertexFiltrationMLP(nn.Module):
    """
    MLP that maps k-nearest neighbor distances to a single filtration value.

    Uses lazy initialization - input dimension is automatically inferred
    on the first forward pass based on k-nearest neighbors.

    Args:
        hidden_dims: List of hidden layer dimensions.
                    None or [] for linear (no hidden layers)
                    [h1] for 1 hidden layer, [h1, h2] for 2 hidden layers, etc.
        dropout: Dropout probability (default 0.0, no dropout)
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.dropout_prob = dropout
        self.output_dim = 1  # Always output 1 filtration value per vertex

        # Build network
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Build MLP network with lazy first layer."""
        layers = []

        if not self.hidden_dims:
            # Linear (no hidden layers): input → 1
            layers.append(nn.LazyLinear(self.output_dim))
        else:
            # First layer is lazy (input dimension inferred on first forward)
            layers.append(nn.LazyLinear(self.hidden_dims[0]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))

            # Middle layers (dimensions known)
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))

            # Final layer → 1 output
            layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, knn_distances: torch.Tensor) -> torch.Tensor:
        """
        Compute vertex filtration values from kNN distances.

        Args:
            knn_distances: Tensor of shape (n_points, k) or (batch, n_points, k)

        Returns:
            Vertex filtration values of shape (n_points,) or (batch, n_points)
        """
        # Apply MLP and squeeze last dimension
        return self.network(knn_distances).squeeze(-1)


class LearnablePointFiltration(nn.Module):
    """
    Learnable filtration for point clouds using flag complex.

    This layer:
    1. Computes k-nearest neighbor distances for each point
    2. Applies an MLP to get vertex filtration values
    3. Computes edge filtrations as: (d^p + fmax^p)^(1/p)
       where d = edge length, fmax = max(f(u), f(v))
    4. Builds simplex tree and computes persistence
    5. Extracts diagrams differentiably via torch.gather

    Args:
        k: Number of nearest neighbors for vertex filtration
        hidden_dims: List of hidden layer dimensions for MLP.
                    None or [] for linear (no hidden layers)
                    [h1] for 1 hidden layer, [h1, h2] for 2 hidden layers, etc.
        dropout: Dropout probability for MLP (default 0.0)
        homology_dimensions: List of homology dimensions to compute
        complex_type: 'alpha' or 'rips'
        max_edge: Maximum edge length for rips complex
        p: Parameter for edge filtration formula (default: 1.0)
        show_progress: Show progress bar during computation
    """

    def __init__(
        self,
        k: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        homology_dimensions: List[int] = [0, 1],
        complex_type: Literal['alpha', 'rips'] = 'alpha',
        max_edge: float = np.inf,
        p: float = 1.0,
        show_progress: bool = False,
    ):
        super().__init__()

        self.k = k
        self.homology_dimensions = homology_dimensions
        self.max_hom_dim = max(homology_dimensions)
        self.complex_type = complex_type
        self.max_edge = max_edge
        self.p = p
        self.show_progress = show_progress

        # Learnable MLP for vertex filtration
        self.mlp = VertexFiltrationMLP(hidden_dims=hidden_dims, dropout=dropout)

    def compute_knn_distances(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Compute k-nearest neighbor distances for each point.

        Args:
            pts: Point cloud of shape (n_points, d)

        Returns:
            kNN distances of shape (n_points, k)
        """
        # Compute pairwise distances
        dists = torch.cdist(pts, pts)  # (n_points, n_points)

        # Get k+1 smallest (including self with distance 0)
        # Then exclude self (first column)
        k_plus_1 = min(self.k + 1, pts.shape[0])
        knn_dists, _ = torch.topk(dists, k_plus_1, dim=-1, largest=False)

        # Exclude self-distance (first column) and take k neighbors
        knn_dists = knn_dists[:, 1:self.k + 1]

        # Pad with zeros if fewer than k neighbors
        if knn_dists.shape[1] < self.k:
            pad_size = self.k - knn_dists.shape[1]
            knn_dists = torch.nn.functional.pad(knn_dists, (0, pad_size), value=0.0)

        return knn_dists

    def compute_edge_filtration(
        self,
        pts: torch.Tensor,
        vertex_filts: torch.Tensor,
        edges: np.ndarray
    ) -> torch.Tensor:
        """
        Compute edge filtration values.

        Edge filtration = (d^p + fmax^p)^(1/p)
        where d = edge length, fmax = max(f(u), f(v))

        When p=1: edge_filt = d + max(f(u), f(v))

        Args:
            pts: Point cloud of shape (n_points, d)
            vertex_filts: Vertex filtration values of shape (n_points,)
            edges: Edge indices of shape (n_edges, 2)

        Returns:
            Edge filtration values of shape (n_edges,)
        """
        p = self.p

        # Get vertex filtration values for edge endpoints
        edges_tensor = torch.from_numpy(edges).long().to(pts.device)
        f_u = vertex_filts[edges_tensor[:, 0]]
        f_v = vertex_filts[edges_tensor[:, 1]]
        fmax = torch.maximum(f_u, f_v)

        # Compute edge lengths
        pts_u = pts[edges_tensor[:, 0]]
        pts_v = pts[edges_tensor[:, 1]]
        d = torch.linalg.norm(pts_u - pts_v, dim=-1)

        # Edge filtration formula
        if p == 1.0:
            return d + fmax
        else:
            return (d ** p + fmax ** p) ** (1 / p)

    def get_simplex_tree(self, pts_np: np.ndarray) -> gudhi.SimplexTree:
        """
        Create simplex tree from point cloud.

        Args:
            pts_np: Point cloud as numpy array of shape (n_points, d)

        Returns:
            GUDHI SimplexTree
        """
        if self.complex_type == 'alpha':
            alpha = gudhi.AlphaComplex(points=pts_np)
            st = alpha.create_simplex_tree(default_filtration_value=False)
        elif self.complex_type == 'rips':
            rips = gudhi.RipsComplex(points=pts_np, max_edge_length=self.max_edge)
            st = rips.create_simplex_tree(max_dimension=2)
        else:
            raise ValueError(f"Unknown complex_type: {self.complex_type}")

        return st

    def update_filtration(
        self,
        st: gudhi.SimplexTree,
        vertex_filts_np: np.ndarray,
        edge_filts_np: np.ndarray,
        edges: np.ndarray
    ) -> gudhi.SimplexTree:
        """
        Update simplex tree with learned filtration values.

        Args:
            st: Original simplex tree
            vertex_filts_np: Vertex filtration values
            edge_filts_np: Edge filtration values
            edges: Edge indices

        Returns:
            New simplex tree with updated filtrations
        """
        # Create new simplex tree with learned filtrations
        new_st = gudhi.SimplexTree()

        # Insert vertices with learned filtrations
        for i, filt in enumerate(vertex_filts_np):
            new_st.insert([i], filtration=float(filt))

        # Insert edges with learned filtrations
        for (i, j), filt in zip(edges, edge_filts_np):
            new_st.insert([i, j], filtration=float(filt))

        # Expand to higher simplices (st-expansion)
        new_st.expansion(self.max_hom_dim + 1)

        # Ensure filtration is non-decreasing
        new_st.make_filtration_non_decreasing()

        return new_st

    def extract_diagrams_single(
        self,
        pts: torch.Tensor,
        vertex_filts: torch.Tensor,
        edge_filts: torch.Tensor,
        edges: np.ndarray,
        pers_generators: tuple
    ) -> List[torch.Tensor]:
        """
        Extract persistence diagrams differentiably for a single point cloud.

        Uses torch.gather to extract birth/death values from filtrations,
        enabling gradient flow.

        Args:
            pts: Point cloud of shape (n_points, d)
            vertex_filts: Vertex filtration values of shape (n_points,)
            edge_filts: Edge filtration values of shape (n_edges,)
            edges: Edge indices of shape (n_edges, 2)
            pers_generators: Output from st.flag_persistence_generators()

        Returns:
            List of diagrams, one per homology dimension
        """
        diagrams = []

        # Create edge index lookup: (i,j) -> edge_index
        edge_to_idx = {}
        for idx, (i, j) in enumerate(edges):
            edge_to_idx[(min(i, j), max(i, j))] = idx

        # H0 generators: (birth_vertex, death_edge)
        # pers_generators[0] has shape (n_pairs, 3): [birth_v, death_v1, death_v2]
        if 0 in self.homology_dimensions:
            h0_gens = pers_generators[0]
            if len(h0_gens) > 0:
                h0_gens = np.array(h0_gens)
                # Birth: vertex filtration
                birth_indices = torch.from_numpy(h0_gens[:, 0]).long().to(pts.device)
                births = vertex_filts[birth_indices]

                # Death: edge filtration
                death_edges = h0_gens[:, 1:]
                death_indices = torch.tensor([
                    edge_to_idx[(min(e[0], e[1]), max(e[0], e[1]))]
                    for e in death_edges
                ], device=pts.device)
                deaths = edge_filts[death_indices]

                dgm_h0 = torch.stack([births, deaths], dim=-1)
            else:
                dgm_h0 = torch.zeros((0, 2), device=pts.device)
            diagrams.append(dgm_h0)

        # H1 generators: (birth_edge, death_edge)
        # pers_generators[1][0] has shape (n_pairs, 4): [birth_v1, birth_v2, death_v1, death_v2]
        if 1 in self.homology_dimensions:
            if len(pers_generators) > 1 and len(pers_generators[1]) > 0:
                h1_gens = np.array(pers_generators[1][0]) if len(pers_generators[1][0]) > 0 else np.array([]).reshape(0, 4)
                if len(h1_gens) > 0:
                    # Birth: edge filtration
                    birth_edges = h1_gens[:, :2]
                    birth_indices = torch.tensor([
                        edge_to_idx[(min(e[0], e[1]), max(e[0], e[1]))]
                        for e in birth_edges
                    ], device=pts.device)
                    births = edge_filts[birth_indices]

                    # Death: edge filtration
                    death_edges = h1_gens[:, 2:]
                    death_indices = torch.tensor([
                        edge_to_idx[(min(e[0], e[1]), max(e[0], e[1]))]
                        for e in death_edges
                    ], device=pts.device)
                    deaths = edge_filts[death_indices]

                    dgm_h1 = torch.stack([births, deaths], dim=-1)
                else:
                    dgm_h1 = torch.zeros((0, 2), device=pts.device)
            else:
                dgm_h1 = torch.zeros((0, 2), device=pts.device)
            diagrams.append(dgm_h1)

        return diagrams

    def forward_single(self, pts: torch.Tensor) -> List[torch.Tensor]:
        """
        Process a single point cloud.

        Args:
            pts: Point cloud of shape (n_points, d)

        Returns:
            List of persistence diagrams (one per homology dimension)
        """
        device = pts.device
        pts_np = pts.detach().cpu().numpy()

        # 1. Compute kNN distances
        knn_dists = self.compute_knn_distances(pts)

        # 2. Apply MLP to get vertex filtrations
        vertex_filts = self.mlp(knn_dists)

        # 3. Get simplex tree and extract edges
        st = self.get_simplex_tree(pts_np)
        edges = np.array([s[0] for s in st.get_skeleton(1) if len(s[0]) == 2])

        if len(edges) == 0:
            # No edges - return empty diagrams
            return [torch.zeros((0, 2), device=device) for _ in self.homology_dimensions]

        # 4. Compute edge filtrations
        edge_filts = self.compute_edge_filtration(pts, vertex_filts, edges)

        # 5. Update simplex tree with learned filtrations
        vertex_filts_np = vertex_filts.detach().cpu().numpy()
        edge_filts_np = edge_filts.detach().cpu().numpy()
        new_st = self.update_filtration(st, vertex_filts_np, edge_filts_np, edges)

        # 6. Compute persistence
        new_st.compute_persistence()
        pers_generators = new_st.flag_persistence_generators()

        # 7. Extract diagrams differentiably
        diagrams = self.extract_diagrams_single(
            pts, vertex_filts, edge_filts, edges, pers_generators
        )

        return diagrams

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams for batch of point clouds.

        Args:
            X: Input of shape (n_points, d) for single point cloud or
               (n_samples, n_points, d) for batch of point clouds.

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2)
        """
        # Handle single sample vs batch
        if X.ndim == 2:
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        n_samples = X.shape[0]

        # Initialize output: [hom_dim][sample_idx] -> diagram
        all_diagrams = [[] for _ in self.homology_dimensions]

        # Process each sample
        iterator = tqdm(range(n_samples), desc="Computing Flag Complex") if self.show_progress else range(n_samples)
        for i in iterator:
            sample_diagrams = self.forward_single(X[i])

            for dim_idx, dgm in enumerate(sample_diagrams):
                all_diagrams[dim_idx].append(dgm)

        return all_diagrams

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)