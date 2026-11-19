# Mini-Projeto: Buscas Clássicas e Otimização Contínua

Projeto desenvolvido para a disciplina de Inteligência Artificial, com o objetivo de implementar e avaliar algoritmos de busca e otimização aplicados à transformação de sequências binárias.

## Algoritmos implementados

- **BFS (Busca em Largura):** busca a sequência de operações de menor custo para alcançar o estado desejado.
- **Greedy Best-First:** utiliza uma heurística baseada na quantidade de blocos consecutivos de diferenças.
- **Greedy otimizada:** identifica diretamente os blocos que precisam ser invertidos.
- **Gradiente Descendente:** realiza a otimização contínua, considerando o erro em relação ao alvo e uma função de custo para transformações não consecutivas.

## Organização do projeto

- `main.py`: execução do programa.
- `modulos/operacoes.py`: operações com sequências binárias.
- `modulos/buscas.py`: algoritmos de busca.
- `modulos/gradiente.py`: gradiente descendente.
- `modulos/experimentos.py`: experimentos e geração de gráficos.
- `modulos/testes.py`: testes de correção.
- `modulos/demonstracao.py`: demonstração do exemplo proposto.
- `resultados/`: gráficos e dados experimentais.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o projeto completo:

```bash
python main.py --todos
```

Também é possível executar as etapas individualmente:

```bash
python main.py --demo
python main.py --testes
python main.py --experimentos
```

## Relatório

A descrição da implementação, a fundamentação matemática e os resultados experimentais estão disponíveis no arquivo `2026_Relatório_LuanDinizMazaroRodovalho.pdf`.

**Autor:** Luan Diniz Mazaro Rodovalho
