# Mini-Projeto: Buscas Clássicas e Otimização Contínua

Implementação **modularizada** de BFS, Greedy Best-First, busca gulosa otimizada e gradiente descendente projetado para transformação de sequências de bits. Inclui experimentos, gráficos, relatório LaTeX/PDF e testes. A implementação dos algoritmos foi preservada em relação à versão original; a mudança central foi a organização por responsabilidade.

## Organização

```text
mini_projeto_modularizado/
├── main.py                 # CLI: importa e executa os outros módulos
├── mini_projeto.py         # Atalho de compatibilidade com o comando antigo
├── modulos/
│   ├── __init__.py
│   ├── config.py           # Constantes, exemplo e tamanhos experimentais
│   ├── operacoes.py        # Flips, diferenças, blocos e aplicação do caminho
│   ├── buscas.py           # BFS, Greedy Best-First, Greedy otimizada
│   ├── gradiente.py        # Função erro, regularização e descida projetada
│   ├── demonstracao.py     # Execução passo a passo do exemplo
│   ├── experimentos.py    # Benchmark, CSVs e gráficos
│   └── testes.py           # Testes de correção e gradiente numérico
├── resultados/             # CSVs e figuras dos experimentos
├── requirements.txt
├── relatorio_revisado.tex
└── relatorio_revisado.pdf
```

## Dependências

- Python 3.10 ou mais recente;
- `numpy` e `matplotlib` (`pip install -r requirements.txt`);
- opcional: `pdflatex` para recompilar o relatório.

## Como executar

Na raiz do projeto, utilize:

```bash
pip install -r requirements.txt
python main.py --testes
python main.py --demo
python main.py --experimentos --n 30 --seed 2026 --saida resultados
# ou tudo em sequência:
python main.py --todos
```

**Compatibilidade:** os comandos antigos `python mini_projeto.py --todos` e similares continuam funcionando; esse arquivo apenas encaminha para `main.main()`.

## Interpretação dos experimentos

Os intervalos de inversão são inclusivos e usam índices iniciados em 1. A BFS convencional e a Greedy Best-First explícita enumeram até `L(L+1)/2` vizinhos por estado e, por custo computacional, foram limitadas a `L <= 16` no benchmark. Para os tamanhos `L = 2, 4, ..., 4096`, foi executada a variante gulosa otimizada, que utiliza diretamente os blocos de diferenças. Campos vazios nos CSVs representam algoritmos não executados, não medições iguais a zero.

O gradiente descendente utiliza a penalização diferenciável de adjacência `C(z)` e registra os vetores de transformação entre as iterações. Veja a fundamentação no relatório.

Os arquivos gerados incluem `experimentos_brutos.csv`, `resumo.csv`, `gradiente_lambda.csv` e cinco gráficos. Tempos de execução podem variar entre máquinas.

## Verificações

O comando `python main.py --testes` verifica, entre outros pontos, todos os 256 pares de estados de quatro bits e compara o gradiente analítico com diferenças finitas. O exemplo do enunciado tem custo ótimo de cinco inversões.

## Relatório e repositório

O PDF e o LaTeX estão incluídos. A execução indicada no relatório foi atualizada para `python main.py --todos`.

Repositório: https://github.com/LuanDinizMR/mini-projeto-buscas-otimizacao-IA-0

Para publicar a alteração no GitHub após substituir os arquivos no seu diretório local:

```bash
git add .
git commit -m "Modulariza implementacao dos algoritmos"
git push
```
