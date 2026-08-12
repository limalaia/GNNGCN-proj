# Melhores configurações dos modelos

**Função peso exponencial:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 4096         | 2               | 0.0001        | 181.06006                 |
| 08/02/2025 | Dia com zero chuva  | 32           | 2               | 0.001         | 0.032061                  |
| 12/01/2025 | Dia com pouca chuva | 64           | 2               | 0.001         | 0.1454                    |

![Expo 1](../Markdown_Images/expo1.png)
![Expo 2](../Markdown_Images/expo2.png)

**Inverso da distância:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 256         | 2               | 0.001        | 213.126129                 |
| 08/02/2025 | Dia com zero chuva  | 32           | 2               | 0.001         | 0.033561                |
| 12/01/2025 | Dia com pouca chuva | 32           | 2               | 0.0001         | 0.180893                    |

![Inv Dis 1](../Markdown_Images/inv_dis1.png)
![Inv Dis 2](../Markdown_Images/inv_dis2.png)

**Inverso da distância ao quadrado:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 2048         | 2               | 0.001        | 247.250687                 |
| 08/02/2025 | Dia com zero chuva  | 32           | 2               | 0.001         | 0.032037                |
| 12/01/2025 | Dia com pouca chuva | 32           | 2               | 0.0001         | 0.178934                    |

![Inv Dis Quad 1](../Markdown_Images/inv_dis_quad1.png)
![Inv Dis Quad 2](../Markdown_Images/inv_dis_quad2.png)

**Gaussiana com self-tuning:**

| Data       | Cenário             | n_hidden_dim | n_hidden_layers | learning_rate | MSE médio de todos os nós |
| ---------- | ------------------- | ------------ | --------------- | ------------- | ------------------------- |
| 12/05/2024 | Dia com mais chuva  | 512         | 2               | 0.001        | 173.424149                 |
| 08/02/2025 | Dia com zero chuva  | 8           | 2               | 0.001         | 0.028943                  |
| 12/01/2025 | Dia com pouca chuva | 512           | 2               | 0.001         | 0.151706                    |