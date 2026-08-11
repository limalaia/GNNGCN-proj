import json

# Lê o notebook
with open('c:\\Users\\loren\\GNNGCN-proj\\Notebooks\\WeightedEdges_compact_gaussian.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encontra o índice da célula com build_weighted_adjacency_selftuning
target_idx = -1
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source_str = ''.join(cell.get('source', []))
        if 'def build_weighted_adjacency_selftuning' in source_str:
            target_idx = idx
            print(f"Encontrada célula build_weighted_adjacency_selftuning no índice {idx}")
            break

if target_idx > 0:
    # Cria célula code com exemplo de uso
    example_cell = {
        "cell_type": "code",
        "metadata": {"language": "python"},
        "source": [
            "# EXEMPLO: Comparar Gaussiano com Sigma Global vs Self-Tuning\n",
            "# =============================================================\n",
            "\n",
            "# Método 1: Gaussiano com sigma global = 1.0\n",
            "adjacency_global, adjacency_norm_global = build_weighted_adjacency(\n",
            "    coords, n_neighbors=61, sigma=1.0\n",
            ")\n",
            "print(f\"✓ Gaussiano Global: sigma=1.0\")\n",
            "print(f\"  Peso mínimo: {adjacency_global[adjacency_global > 0].min():.6f}\")\n",
            "print(f\"  Peso máximo: {adjacency_global[adjacency_global > 0].max():.6f}\")\n",
            "\n",
            "# Método 2: Self-Tuning Spectral Clustering\n",
            "adjacency_st, adjacency_norm_st, sigma_array_st = build_weighted_adjacency_selftuning(\n",
            "    coords, n_neighbors=61\n",
            ")\n",
            "print(f\"\\n✓ Self-Tuning Spectral Clustering:\")\n",
            "print(f\"  Sigma mínimo: {sigma_array_st.min():.6f}\")\n",
            "print(f\"  Sigma médio: {sigma_array_st.mean():.6f}\")\n",
            "print(f\"  Sigma máximo: {sigma_array_st.max():.6f}\")\n",
            "print(f\"  Peso mínimo: {adjacency_st[adjacency_st > 0].min():.6f}\")\n",
            "print(f\"  Peso máximo: {adjacency_st[adjacency_st > 0].max():.6f}\")\n",
            "\n",
            "print(\"\\n✓ Ambos os métodos estão prontos para treinamento!\")\n"
        ]
    }
    
    # Insere a célula de exemplo após build_weighted_adjacency_selftuning
    nb['cells'].insert(target_idx + 1, example_cell)
    
    print(f"✓ Adicionada célula de exemplo após build_weighted_adjacency_selftuning")
else:
    print("⚠ Célula build_weighted_adjacency_selftuning não encontrada!")
    exit(1)

# Salva o notebook
with open('c:\\Users\\loren\\GNNGCN-proj\\Notebooks\\WeightedEdges_compact_gaussian.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("✓ Notebook atualizado com célula de exemplo!")
