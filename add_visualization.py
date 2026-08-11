import json

# Lê o notebook
with open('c:\\Users\\loren\\GNNGCN-proj\\Notebooks\\WeightedEdges_compact_gaussian.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encontra a última célula code
last_code_idx = -1
for idx in range(len(nb['cells']) - 1, -1, -1):
    if nb['cells'][idx].get('cell_type') == 'code':
        last_code_idx = idx
        break

if last_code_idx > 0:
    # Cria célula markdown
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Visualização: Distribuição de Sigmas no Self-Tuning\n",
            "\n",
            "O mapa abaixo mostra como cada estação obtém um $\\sigma_i$ adaptado à sua densidade local de vizinhos.\n"
        ]
    }
    
    # Cria célula code para visualização
    viz_cell = {
        "cell_type": "code",
        "metadata": {"language": "python"},
        "source": [
            "# Visualização da distribuição de sigmas no self-tuning\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
            "\n",
            "# Painel 1: Histograma de sigmas\n",
            "axes[0].hist(sigma_array_st, bins=15, color='steelblue', edgecolor='black', alpha=0.7)\n",
            "axes[0].set_xlabel('Sigma ($\\\\sigma_i$)', fontsize=12)\n",
            "axes[0].set_ylabel('Frequência', fontsize=12)\n",
            "axes[0].set_title('Distribuição de Sigmas Adaptativos\\n(Self-Tuning)', fontsize=13, fontweight='bold')\n",
            "axes[0].grid(axis='y', alpha=0.3)\n",
            "\n",
            "# Painel 2: Mapa geográfico com sigmas por cor\n",
            "scatter = axes[1].scatter(\n",
            "    coords[:, 0], coords[:, 1],\n",
            "    c=sigma_array_st, cmap='RdYlBu_r', s=100, edgecolors='black', linewidth=0.5\n",
            ")\n",
            "axes[1].set_xlabel('Longitude', fontsize=12)\n",
            "axes[1].set_ylabel('Latitude', fontsize=12)\n",
            "axes[1].set_title('Mapa Geográfico com Sigmas\\n(Cores: vermelho=sigma alto, azul=sigma baixo)', fontsize=13, fontweight='bold')\n",
            "cbar = plt.colorbar(scatter, ax=axes[1])\n",
            "cbar.set_label('$\\\\sigma_i$ (Self-Tuning)', fontsize=11)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\n📊 Análise de Self-Tuning Sigmas:\")\n",
            "print(f\"   Σmin = {sigma_array_st.min():.4f} (regiões densas - vizinhos próximos)\")\n",
            "print(f\"   Σmédio = {sigma_array_st.mean():.4f}\")\n",
            "print(f\"   Σmax = {sigma_array_st.max():.4f} (regiões esparsas - vizinhos afastados)\")\n",
            "print(f\"   Variância = {sigma_array_st.std():.4f}\")\n"
        ]
    }
    
    # Insere células no final do notebook
    nb['cells'].append(md_cell)
    nb['cells'].append(viz_cell)
    
    print(f"✓ Adicionadas 2 células (markdown + visualização) no final do notebook")
else:
    print("⚠ Nenhuma célula code encontrada!")
    exit(1)

# Salva o notebook
with open('c:\\Users\\loren\\GNNGCN-proj\\Notebooks\\WeightedEdges_compact_gaussian.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("✓ Notebook atualizado com visualização de sigmas!")
