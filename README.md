# GNNGCN-proj

Repositório desenvolvido para o projeto **Previsão de Desastres Climáticos Utilizando Redes Neurais em Grafos**, com foco na modelagem de dados meteorológicos por meio de **Redes Neurais em Grafos**, especialmente modelos baseados em **GCN** e arquiteturas espaço-temporais.

O objetivo principal é estudar a previsão de variáveis climáticas, como precipitação, a partir de dados meteorológicos organizados em uma estrutura de grafo, onde cada nó representa uma estação meteorológica e as arestas representam relações espaciais entre estações próximas.

---

## Objetivo do projeto

Este projeto busca aplicar modelos de aprendizado profundo em grafos para previsão climática, explorando a dependência espacial entre estações meteorológicas.

Em linhas gerais, o projeto envolve:

- construção de grafos a partir da localização geográfica das estações;
- extração e pré-processamento de variáveis meteorológicas;
- criação de janelas temporais para previsão;
- treinamento de modelos baseados em redes neurais em grafos;
- avaliação das previsões por métricas de erro e gráficos comparativos.

---

## Estrutura do repositório

```text
GNNGCN-proj/
│
├── Datasets/          # Dados meteorológicos utilizados no projeto
├── Grafos/            # Imagens dos grafos construídos
├── Graphics/          # Gráficos gerados durante os experimentos
├── Notebooks/         # Notebooks principais de execução e análise
├── Results/           # Resultados dos modelos treinados
├── Tables/            # Tabelas com resultados e métricas
├── Tables_Loss/       # Tabelas relacionadas à função de perda
├── src/               # Código-fonte modularizado
│   ├── Data/          # Funções de carregamento, preparação e pré-processamento
│   ├── Evaluation/    # Métricas e gráficos de avaliação
│   ├── Graph/         # Funções para construção dos grafos
│   ├── Models/        # Definição dos modelos neurais
│   └── Training/      # Rotinas de treinamento e validação
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Pipeline de experimento Weighted GNN

O arquivo `run_experiment.py`, na raiz do projeto, executa uma configuração
específica da GNN ponderada inspirada em `Notebooks/WeightedEdges.ipynb`.
Os parâmetros ficam no início do arquivo, em uma seção com nomes em letras
maiúsculas. Edite esses valores e execute o script sem argumentos:

O arquivo da raiz funciona somente como configuração e ponto de entrada. A
implementação do treinamento, avaliação, geração de artefatos e relatório fica
em `src/experiment_pipeline.py`.

```powershell
python run_experiment.py
```

As funções de perda disponíveis são `mse`, `mae`, `huber` e `weighted_mse`.

Para comparar as execuções concluídas em `OUTPUT_ROOT`, defina no início do
script, por exemplo:

```python
COMPARISON_REPORT = True
PIVOT_PARAMETER = "HIDDEN_DIM"
```

Ao executar uma nova run, o modo de comparação gera `comparison_report.tex` e
`comparison_metrics.png` em `OUTPUT_ROOT`. O relatório contém as métricas de
teste RMSE, MAE e R2 de cada run e três gráficos lado a lado usando o parâmetro
informado como eixo x.

Os parâmetros de execução também podem receber listas. Nesse caso, o script
executa uma run para cada combinação dos valores informados:

```python
HIDDEN_DIM = [32, 64, 128, 256]
EPOCHS = [500, 1000]
PIVOT_PARAMETER = "HIDDEN_DIM"
```

O exemplo acima gera oito runs (`4 x 2`).

Cada execução em lote cria uma pasta em
`outputs/run_TARGET_DATE_HHhMM/`, por exemplo
`outputs/run_2024-05-01_10h39/`. Dentro dela ficam as subpastas das runs de
cada combinação dos parâmetros. Se já houver uma execução no mesmo minuto, o
pipeline acrescenta um sufixo numérico para não sobrescrever resultados.
Quando `COMPARISON_REPORT = True`, a raiz do lote contém somente
`comparison_report.tex`, com a tabela e os três gráficos embutidos no LaTeX.
O relatório também possui uma página com os gráficos de dispersão
`prediction_scatter.png` e os gráficos XY por estação (`predictions_by_node.png`)
de cada run, organizados lado a lado.
Também é incluído um grafo residual por run: a cor branco-vermelho representa
`|real - predito|` em mm e cada rótulo mostra o resíduo assinado da estação.
Cada subpasta de run contém a configuração e o resumo em JSON, métricas por época,
predições e pesos das arestas em CSV, o checkpoint `model.pt` e os seguintes
gráficos:

- curvas de loss de treino, validação e teste;
- grafo KNN geográfico com os splits dos nós;
- grafo residual por estação, em mm, com escala de branco a vermelho;
- mapas de calor das matrizes de adjacência;
- dispersão da precipitação real contra a predita;
- precipitação real e predita por estação.
