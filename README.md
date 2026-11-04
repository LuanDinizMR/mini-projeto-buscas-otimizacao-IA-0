# Mini-projeto: buscas clássicas e otimização

Solução integral do enunciado de 25/08/2026. Código em Python, relatório em LaTeX/PDF e experimentos reproduzíveis.

## Dependências

- Python 3.10+;
- `numpy` e `matplotlib` (`pip install -r requirements.txt`);
- opcional: `pdflatex` para compilar o relatório.

## Executar

```bash
pip install -r requirements.txt
python mini_projeto.py --testes
python mini_projeto.py --demo
python mini_projeto.py --experimentos --n 30 --seed 2026 --saida resultados
# ou simplesmente:
python mini_projeto.py --todos
```

A pasta `resultados/` inclui os dados de cada rodada (`experimentos_brutos.csv`), médias (`resumo.csv`), teste de lambdas (`gradiente_lambda.csv`) e cinco gráficos. A opção `--todos` refaz os arquivos; pequenas diferenças de tempo são esperadas em outra máquina.

## Método de busca

Todos os flips são intervalos inclusivos `[inicio, fim]` usando **posições baseadas em 1**. BFS convencional e Greedy Best-First explícita enumeram todos os L(L+1)/2 intervalos a partir de cada estado. A BFS expande até 2^L estados e **não deve** ser executada em sequências muito longas. No benchmark, as duas buscas explícitas foram limitadas a L <= 16, e os campos ausentes de BFS/Gulosa no CSV indicam **não executado**, não zero.

A função `greedy_otimizado` usa a mesma heurística (número de blocos consecutivos de diferenças) e constrói diretamente uma solução ótima em O(L), permitindo comparação de custo em comprimentos até 4096. `expandidos_otimizado` é uma contagem lógica de etapas, não uma contagem diretamente comparável aos estados enumerados da BFS ou Best-First.

## Método do gradiente

`gradiente_descendente` retorna `ResultadoGD`, incluindo o vetor final `x`, os vetores de transformação `passos` (deltas completos de todas as iterações), a série do objetivo, o passo η e o número de iterações. O gradiente usa a penalidade C(z) = soma das diferenças quadráticas de pesos de flip adjacentes com bordas zero. Há projeção de x no intervalo [0,1]. A derivação está em `relatorio.tex`.

## Publicação no Git

A URL de repositório requerida pelo enunciado deve ser preenchida no final de `relatorio.tex` **depois de publicar este diretório em um repositório Git real**. Não foi criado um repositório remoto automaticamente.

Exemplo, substituindo a URL pela do seu repositório:

```bash
git init
git add .
git commit -m "Implementa buscas e otimização do mini-projeto"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
git push -u origin main
```

## Verificações

`python mini_projeto.py --testes` compara os custos de BFS, gulosa e otimizada para **todos os 256 pares** de estados possíveis de comprimento quatro e verifica o gradiente analítico contra diferenças finitas. O caso do enunciado tem custo ótimo cinco.
