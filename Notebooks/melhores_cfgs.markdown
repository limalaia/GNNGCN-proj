# Melhores configurações dos modelos

**Função peso exponencial:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 4096         | 2               | 0.0001        | 181.06006                 |
| 08/02/2025 | Dia com zero chuva  | 32           | 2               | 0.001         | 0.032061                  |
| 12/01/2025 | Dia com pouca chuva | 64           | 2               | 0.001         | 0.1454                    |

**Gaussiana com self-tuning:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 512         | 2               | 0.001        | 173.424149                 |
| 08/02/2025 | Dia com zero chuva  | 8           | 2               | 0.001         | 0.028943                  |
| 12/01/2025 | Dia com pouca chuva | 512           | 2               | 0.001         | 0.151706                    |

**Inverso da distância:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 256         | 2               | 0.001        | 213.126129                 |
| 08/02/2025 | Dia com zero chuva  | 32           | 2               | 0.001         | 0.033561                |
| 12/01/2025 | Dia com pouca chuva | 32           | 2               | 0.0001         | 0.180893                    |