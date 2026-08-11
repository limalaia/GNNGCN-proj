import json
from pathlib import Path

notebook_path = Path(r'c:\Users\loren\GNNGCN-proj\Notebooks\WeightedEdges_compact_inversodadistancia.ipynb')
with notebook_path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    joined = ''.join(source)
    if 'adjacency_weighted[adjacency_binary] = torch.exp(' in joined:
        new_lines = []
        for line in source:
            if line.strip() == 'adjacency_weighted[adjacency_binary] = torch.exp(':
                new_lines.append('    adjacency_weighted[adjacency_binary] = 1.0 / distances[adjacency_binary]\n')
            elif line.strip() == '-distances[adjacency_binary]' or line.strip() == '    -distances[adjacency_binary]':
                continue
            elif line.strip() == ')' and new_lines and new_lines[-1].startswith('    adjacency_weighted[adjacency_binary] = 1.0 / distances'):
                continue
            else:
                new_lines.append(line)
        # insert inf guard if not already present
        insert_idx = None
        for idx, line in enumerate(new_lines):
            if 'adjacency_weighted.fill_diagonal_(1.0)' in line:
                insert_idx = idx
                break
        if insert_idx is not None:
            if 'adjacency_weighted[adjacency_weighted == float(\'inf\')] = 0.0\n' not in new_lines:
                new_lines.insert(insert_idx, '    adjacency_weighted[adjacency_weighted == float(\'inf\')] = 0.0\n')
        cell['source'] = new_lines
        updated = True
    if 'Regra de peso da grade: w_ij = exp(-||x_i - x_j||_2)' in joined:
        cell['source'] = ["print(\"Regra de peso da grade: w_ij = 1 / ||x_i - x_j||_2\")\n"]
        updated = True

if not updated:
    raise RuntimeError('Nenhuma célula modificada; o padrão esperado não foi encontrado.')

with notebook_path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print('Notebook atualizado para pesos 1/||x_i - x_j||.')
