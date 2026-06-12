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