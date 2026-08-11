import json

# Lê o notebook
with open('c:\\Users\\loren\\GNNGCN-proj\\Notebooks\\WeightedEdges_compact_gaussian.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encontra o índice da célula com build_weighted_adjacency
target_idx = -1
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source_str = ''.join(cell.get('source', []))
        if 'def build_weighted_adjacency' in source_str and 'sigma=1.0' in source_str:
            target_idx = idx
            print(f"Encontrada célula build_weighted_adjacency no índice {idx}")
            break

if target_idx > 0:
    # Cria célula markdown para explicar self-tuning
    md_cell = {
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": [
            "## 4.1. Self-Tuning Spectral Clustering (Adaptativo)\n",
            "\n",
            "Ao invés de usar um único $\\sigma$ global, o **self-tuning** calcula um $\\sigma_i$ **diferente para cada nó** baseado na sua vizinhança local:\n",
            "\n",
            "$$\\sigma_i = d_{i,k}$$\n",
            "\n",
            "onde $d_{i,k}$ é a distância do nó $i$ até seu k-ésimo vizinho mais próximo.\n",
            "\n",
            "### Vantagem\n",
            "\n",
            "Em regiões com **diferentes densidades** de nós:\n",
            "- Nós em regiões densas: $\\sigma$ pequeno (peso concentrado nos vizinhos próximos)\n",
            "- Nós em regiões esparsas: $\\sigma$ grande (peso mais distribuído)\n",
            "\n",
            "Isso **torna o método robusto** a variações de densidade nos dados.\n",
            "\n",
            "### Referência\n",
            "\n",
            "**Zelnik-Manor, L., & Perona, P. (2004).** \"Self-Tuning Spectral Clustering.\" NIPS.\n"
        ]
    }
    
    # Cria célula code com as funções
    code_cell = {
        "cell_type": "code",
        "metadata": {"language": "python"},
        "source": [
            "def compute_self_tuning_sigma(coordinates, n_neighbors):\n",
            "    \"\"\"\n",
            "    Calcula sigma_i = d_{i,k} para cada nó (Self-Tuning Spectral Clustering).\n",
            "    \n",
            "    Referência: Zelnik-Manor, L., & Perona, P. (2004). \"Self-Tuning Spectral Clustering.\" NIPS.\n",
            "    \n",
            "    Args:\n",
            "        coordinates: array (n_nodes, 2) com coordenadas dos nós\n",
            "        n_neighbors: k para k-vizinhos mais próximos\n",
            "    \n",
            "    Returns:\n",
            "        sigma_array: array (n_nodes,) com sigma_i para cada nó\n",
            "    \"\"\"\n",
            "    coordinates = np.asarray(coordinates, dtype=np.float32)\n",
            "    n_nodes = coordinates.shape[0]\n",
            "    \n",
            "    if not 1 <= n_neighbors < n_nodes:\n",
            "        raise ValueError(f\"k deve estar entre 1 e {n_nodes - 1}.\")\n",
            "    \n",
            "    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)\n",
            "    distances, _ = neighbors.kneighbors(coordinates)\n",
            "    distances = distances[:, 1:]  # Exclui o próprio nó\n",
            "    \n",
            "    # sigma_i = distância para o k-ésimo vizinho\n",
            "    sigma_array = distances[:, -1]  # Último (k-ésimo) vizinho\n",
            "    \n",
            "    return sigma_array\n",
            "\n",
            "\n",
            "def build_weighted_adjacency_selftuning(coordinates, n_neighbors, sigma_array=None):\n",
            "    \"\"\"\n",
            "    Constrói A e D^(-1/2) A D^(-1/2) com kernel Gaussiano com self-tuning.\n",
            "    \n",
            "    Se sigma_array for None, usa sigma_i = d_{i,k} (self-tuning).\n",
            "    Caso contrário, usa os sigmas fornecidos.\n",
            "    \n",
            "    Args:\n",
            "        coordinates: array (n_nodes, 2)\n",
            "        n_neighbors: k para k-vizinhos mais próximos\n",
            "        sigma_array: array (n_nodes,) com sigma para cada nó. Se None, computa self-tuning.\n",
            "    \n",
            "    Returns:\n",
            "        adjacency_weighted: matriz ponderada antes da normalização\n",
            "        adjacency_normalized: matriz ponderada normalizada D^{-1/2}AD^{-1/2}\n",
            "        sigma_array: array de sigmas utilizados\n",
            "    \"\"\"\n",
            "    coordinates = np.asarray(coordinates, dtype=np.float32)\n",
            "    n_nodes = coordinates.shape[0]\n",
            "    \n",
            "    if not 1 <= n_neighbors < n_nodes:\n",
            "        raise ValueError(f\"k deve estar entre 1 e {n_nodes - 1}.\")\n",
            "    \n",
            "    # Computa self-tuning se não fornecido\n",
            "    if sigma_array is None:\n",
            "        sigma_array = compute_self_tuning_sigma(coordinates, n_neighbors)\n",
            "    else:\n",
            "        sigma_array = np.asarray(sigma_array, dtype=np.float32)\n",
            "    \n",
            "    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)\n",
            "    _, neighbor_indices = neighbors.kneighbors(coordinates)\n",
            "    neighbor_indices = neighbor_indices[:, 1:]\n",
            "    \n",
            "    adjacency_binary = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)\n",
            "    for i, indices in enumerate(neighbor_indices):\n",
            "        adjacency_binary[i, torch.as_tensor(indices, dtype=torch.long)] = True\n",
            "    \n",
            "    adjacency_binary = adjacency_binary | adjacency_binary.T\n",
            "    adjacency_binary.fill_diagonal_(False)\n",
            "    \n",
            "    coordinates_tensor = torch.as_tensor(coordinates, dtype=torch.float32)\n",
            "    distances = torch.cdist(coordinates_tensor, coordinates_tensor, p=2)\n",
            "    \n",
            "    # Kernel Gaussiano com sigmas por nó: w_ij = exp(-(d_ij^2) / (2*sigma_i*sigma_j))\n",
            "    # Usa média geométrica: sqrt(sigma_i * sigma_j)\n",
            "    sigma_tensor = torch.as_tensor(sigma_array, dtype=torch.float32)\n",
            "    sigma_product = torch.outer(sigma_tensor, sigma_tensor)  # sigma_i * sigma_j\n",
            "    sigma_geometric_mean = torch.sqrt(sigma_product)  # sqrt(sigma_i * sigma_j)\n",
            "    \n",
            "    adjacency_weighted = torch.zeros_like(distances)\n",
            "    adjacency_weighted[adjacency_binary] = torch.exp(\n",
            "        -(distances[adjacency_binary] ** 2) / (2.0 * sigma_geometric_mean[adjacency_binary] ** 2)\n",
            "    )\n",
            "    adjacency_weighted.fill_diagonal_(1.0)\n",
            "    \n",
            "    degree = adjacency_weighted.sum(dim=1).clamp_min(1e-12)\n",
            "    inv_sqrt_degree = degree.rsqrt()\n",
            "    adjacency_normalized = (\n",
            "        inv_sqrt_degree[:, None]\n",
            "        * adjacency_weighted\n",
            "        * inv_sqrt_degree[None, :]\n",
            "    )\n",
            "    \n",
            "    if not torch.isfinite(adjacency_normalized).all():\n",
            "        raise FloatingPointError(\"A matriz ponderada contém valores não finitos.\")\n",
            "    \n",
            "    return adjacency_weighted, adjacency_normalized, sigma_array\n",
            "\n",
            "\n",
            "print(\"✓ Funções de self-tuning carregadas!\")\n",
            "print(\"  - compute_self_tuning_sigma(coordinates, n_neighbors)\")\n",
            "print(\"  - build_weighted_adjacency_selftuning(coordinates, n_neighbors, sigma_array=None)\")\n"
        ]
    }
    
    # Insere as novas células após build_weighted_adjacency
    nb['cells'].insert(target_idx + 1, md_cell)
    nb['cells'].insert(target_idx + 2, code_cell)
    
    print(f"✓ Adicionadas 2 células (markdown + code) após a função build_weighted_adjacency")
else:
    print("⚠ Célula build_weighted_adjacency não encontrada!")
    exit(1)

# Salva o notebook
with open('c:\\Users\\loren\\GNNGCN-proj\\Notebooks\\WeightedEdges_compact_gaussian.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("✓ Notebook salvo com sucesso!")
