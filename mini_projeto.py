#!/usr/bin/env python3
"""Mini-projeto: BFS, busca gulosa e gradiente descendente.

Uso:
    python mini_projeto.py --demo
    python mini_projeto.py --experimentos --n 30 --seed 2026 --saida resultados
    python mini_projeto.py --todos

Índices de intervalos na interface são inclusivos e começam em 1.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXEMPLO_INICIAL = '00001001000011001100111'
EXEMPLO_FINAL = '00000000000000000000000'
TAMANHOS = [2 ** p for p in range(1, 13)]  # 2, 4, ..., 4096 (potências de 2)


@dataclass
class ResultadoBusca:
    caminho: list[tuple[int, int]]  # intervalos inclusivos, índices 1-based
    custo: int
    expandidos: int
    gerados: int
    segundos: float


def validar(inicial: str, alvo: str) -> None:
    if len(inicial) != len(alvo) or not inicial or set(inicial + alvo) - {'0', '1'}:
        raise ValueError('Estados devem ser strings binárias, não vazias e de mesmo tamanho.')


def flip_bit(estado: str, pos: int) -> str:
    """Inverte um bit; pos é um índice baseado em 1."""
    return flip_intervalo(estado, pos, pos)


def flip_intervalo(estado: str, inicio: int, fim: int) -> str:
    """Inverte todos os bits do intervalo inclusivo [inicio, fim] (1-based)."""
    if not 1 <= inicio <= fim <= len(estado) or set(estado) - {'0', '1'}:
        raise ValueError('Intervalo ou estado inválido.')
    pre = estado[:inicio-1]
    mid = ''.join('1' if c == '0' else '0' for c in estado[inicio-1:fim])
    return pre + mid + estado[fim:]


def diferencas(inicial: str, alvo: str) -> str:
    validar(inicial, alvo)
    return ''.join('1' if x != y else '0' for x, y in zip(inicial, alvo))


def blocos_um(d: str) -> list[tuple[int, int]]:
    """Retorna blocos máximos de 1, em coordenadas 1-based inclusivas."""
    blocos = []
    inicio = None
    for pos, bit in enumerate(d, 1):
        if bit == '1' and inicio is None:
            inicio = pos
        elif bit == '0' and inicio is not None:
            blocos.append((inicio, pos - 1))
            inicio = None
    if inicio is not None:
        blocos.append((inicio, len(d)))
    return blocos


def heuristica(estado: str, alvo: str) -> int:
    return len(blocos_um(diferencas(estado, alvo)))


def mascaras_intervalos(l: int) -> list[tuple[int, tuple[int, int]]]:
    """Um mask por intervalo; string binária corresponde ao formato MSB...LSB."""
    return [(((1 << (fim - ini + 1)) - 1) << (l - fim), (ini, fim))
            for ini in range(1, l+1) for fim in range(ini, l+1)]


def reconstruir(anteriores: dict[int, tuple[int, tuple[int, int]] | None],
                fim: int) -> list[tuple[int, int]]:
    caminho = []
    atual = fim
    while anteriores[atual] is not None:
        anterior, op = anteriores[atual]
        caminho.append(op)
        atual = anterior
    caminho.reverse()
    return caminho


def bfs(inicial: str, alvo: str, max_l: int = 14) -> ResultadoBusca:
    """BFS completa sobre o grafo de sequências; limite protege a memória/tempo.

    Cada aresta corresponde à inversão de qualquer intervalo [i, j], custo 1.
    """
    validar(inicial, alvo)
    l = len(inicial)
    if l > max_l:
        raise ValueError(f'BFS convencional inviável para L={l}; limite configurado: {max_l}.')
    inicio_int, fim_int = int(inicial, 2), int(alvo, 2)
    t0 = time.perf_counter()
    fila = deque([inicio_int])
    pais: dict[int, tuple[int, tuple[int, int]] | None] = {inicio_int: None}
    masks = mascaras_intervalos(l)
    expandidos = gerados = 0
    while fila:
        atual = fila.popleft()
        expandidos += 1
        if atual == fim_int:
            caminho = reconstruir(pais, atual)
            return ResultadoBusca(caminho, len(caminho), expandidos, gerados,
                                  time.perf_counter() - t0)
        for mask, op in masks:
            gerados += 1
            vizinho = atual ^ mask
            if vizinho not in pais:
                pais[vizinho] = atual, op
                fila.append(vizinho)
    raise AssertionError('O espaço de sequências é conexo; alvo deve ser alcançável.')


def greedy_best_first(inicial: str, alvo: str, max_l: int = 14) -> ResultadoBusca:
    """Busca gulosa Best-First explícita: fronteira ordenada apenas por h(n)."""
    validar(inicial, alvo)
    l = len(inicial)
    if l > max_l:
        raise ValueError('Busca gulosa explícita enumeraria O(L²) vizinhos por estado.')
    t0 = time.perf_counter()
    inicio_int, fim_int = int(inicial, 2), int(alvo, 2)
    masks = mascaras_intervalos(l)
    pais: dict[int, tuple[int, tuple[int, int]] | None] = {inicio_int: None}
    heap = [(heuristica(inicial, alvo), 0, inicio_int)]
    contador = 0
    expandidos = gerados = 0
    while heap:
        _, _, atual = heapq.heappop(heap)
        expandidos += 1
        if atual == fim_int:
            caminho = reconstruir(pais, atual)
            return ResultadoBusca(caminho, len(caminho), expandidos, gerados,
                                  time.perf_counter() - t0)
        for mask, op in masks:
            gerados += 1
            vizinho = atual ^ mask
            if vizinho not in pais:
                pais[vizinho] = atual, op
                contador += 1
                bits_vizinho = format(vizinho, f'0{l}b')
                heapq.heappush(heap, (heuristica(bits_vizinho, alvo), contador, vizinho))
    raise AssertionError('Alvo não alcançado.')


def greedy_otimizado(inicial: str, alvo: str) -> ResultadoBusca:
    """Escolhe sempre um bloco de diferenças; mesmo critério h da busca gulosa.

    Como inverter um bloco diminui h exatamente em 1, evita enumerar L(L+1)/2
    vizinhos sem alterar o número ótimo de passos. Complexidade O(L).
    """
    t0 = time.perf_counter()
    d = diferencas(inicial, alvo)
    caminho = blocos_um(d)
    return ResultadoBusca(caminho, len(caminho), len(caminho) + 1,
                          len(caminho), time.perf_counter() - t0)


def aplicar_caminho(inicial: str, caminho: list[tuple[int, int]]) -> str:
    estado = inicial
    for ini, fim in caminho:
        estado = flip_intervalo(estado, ini, fim)
    return estado


def erro(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x-y)**2))


def peso_flip(x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """z_i=x_i se x0_i=0; z_i=1-x_i se x0_i=1; z_i ∈ [0,1]."""
    a = 1 - 2*x0
    return a*(x-x0)


def custo_adjacencia(z: np.ndarray) -> float:
    """C(z)=Σ_i (z_{i+1}-z_i)^2, com z_0=z_{L+1}=0."""
    zw = np.pad(z, (1,1), constant_values=0.)
    return float(np.sum(np.diff(zw)**2))


def objetivo(x: np.ndarray, x0: np.ndarray, y: np.ndarray,
             lam: float) -> float:
    return erro(x, y) + lam * custo_adjacencia(peso_flip(x, x0))


def gradiente(x: np.ndarray, x0: np.ndarray, y: np.ndarray,
              lam: float) -> np.ndarray:
    z = peso_flip(x, x0)
    zw = np.pad(z, (1,1), constant_values=0.)
    laplaciano = 2*z - zw[:-2] - zw[2:]
    a = 1-2*x0
    return 2*(x-y) + 2*lam*a*laplaciano


@dataclass
class ResultadoGD:
    x: np.ndarray
    passos: list[np.ndarray]
    historico: list[float]
    iteracoes: int
    eta: float
    lam: float


def gradiente_descendente(inicial: str, alvo: str, lam: float = 0.0,
                         eta: float | None = None, max_iter: int = 10000,
                         tolerancia: float = 1e-11) -> ResultadoGD:
    """GD projetado: x ← clip(x-η∇J, 0, 1).

    Hessiana = 2I + 2λ A D^T D A, e λmax(D^T D)≤4;
    eta = 0.9/(2+8λ) é conservadora para todas as dimensões.
    Cada passo armazenado é Δx=x_{t+1}-x_t, não um flip discreto.
    """
    validar(inicial, alvo)
    if lam < 0:
        raise ValueError('É necessário λ>=0.')
    x0 = np.fromiter((float(c) for c in inicial), dtype=float)
    y = np.fromiter((float(c) for c in alvo), dtype=float)
    x = x0.copy()
    if eta is None:
        eta = 0.9 / (2 + 8*lam)
    if eta <= 0:
        raise ValueError('η precisa ser positiva.')
    historico = [objetivo(x,x0,y,lam)]
    passos = []
    for it in range(1, max_iter + 1):
        novo_x = np.clip(x - eta*gradiente(x,x0,y,lam), 0., 1.)
        passo = novo_x - x
        passos.append(passo)
        x = novo_x
        historico.append(objetivo(x,x0,y,lam))
        if np.max(np.abs(passo)) < tolerancia:
            return ResultadoGD(x, passos, historico, it, eta, lam)
    return ResultadoGD(x, passos, historico, max_iter, eta, lam)


def bits_discretizados(x: np.ndarray) -> str:
    return ''.join('1' if xi >= 0.5 else '0' for xi in x)


def demo() -> None:
    s, y = EXEMPLO_INICIAL, EXEMPLO_FINAL
    print('\n=== EXEMPLO DO ENUNCIADO ===')
    print('Inicial:', s, '\nAlvo:   ', y)
    r = greedy_otimizado(s,y)
    print('Blocos:', r.caminho, ' | custo ótimo:', r.custo)
    estado = s
    for k, op in enumerate(r.caminho, 1):
        estado = flip_intervalo(estado, *op)
        print(f'{k}. flip {op}: {estado}')
    assert estado == y
    print('\n=== GRADIENTE DESCENDENTE ===')
    for lam in (0, .05, .25, 1., 4.):
        gd = gradiente_descendente(s, y, lam)
        x0 = np.array([int(c) for c in s], dtype=float)
        z = peso_flip(gd.x, x0)
        print(f'lambda={lam:>4}: iter={gd.iteracoes:>5}, L={erro(gd.x,np.zeros(len(s))):.8f}, '
              f'C={custo_adjacencia(z):.8f}, J={gd.historico[-1]:.8f}, '
              f'bits={bits_discretizados(gd.x)}')
    x = np.array([1.0]); y0 = np.array([0.0]); eta = 0.75
    print(f'Contraexemplo sem projeção: x0=1, y=0, eta={eta}, x1={x[0]-eta*2*(x[0]-y0[0]):.2f}')


def salvar_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experimentos(n: int = 30, seed: int = 2026, saida: str = 'resultados',
                 limite_bfs: int = 16) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pasta = Path(saida)
    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    linhas = []
    print(f'\n=== EXPERIMENTOS: N={n} pares/tamanho, seed={seed} ===')
    for l in TAMANHOS:
        for rep in range(n):
            s = ''.join(map(str,rng.integers(0,2,size=l)))
            y = ''.join(map(str,rng.integers(0,2,size=l)))
            referencia = heuristica(s,y)
            bf = bfs(s,y,max_l=limite_bfs) if l <= limite_bfs else None
            gulosa = greedy_best_first(s,y,max_l=limite_bfs) if l <= limite_bfs else None
            rapida = greedy_otimizado(s,y)
            assert aplicar_caminho(s,rapida.caminho) == y
            assert rapida.custo == referencia
            if bf:
                assert aplicar_caminho(s,bf.caminho) == y and bf.custo == referencia
            if gulosa:
                assert aplicar_caminho(s,gulosa.caminho) == y and gulosa.custo == referencia
            linhas.append(dict(L=l, repeticao=rep, custo_otimo=referencia,
                               tempo_bfs_s=bf.segundos if bf else '',
                               expandidos_bfs=bf.expandidos if bf else '',
                               gerados_bfs=bf.gerados if bf else '',
                               custo_bfs=bf.custo if bf else '',
                               tempo_gulosa_s=gulosa.segundos if gulosa else '',
                               expandidos_gulosa=gulosa.expandidos if gulosa else '',
                               gerados_gulosa=gulosa.gerados if gulosa else '',
                               custo_gulosa=gulosa.custo if gulosa else '',
                               tempo_otimizado_s=rapida.segundos,
                               custo_otimizado=rapida.custo,
                               expandidos_otimizado=rapida.expandidos))
        print(f'L={l:4d}: custo médio={np.mean([r["custo_otimo"] for r in linhas if r["L"]==l]):.2f}, '
              f'BFS={"executada" if l<=limite_bfs else "N/A (inviável)"}')
    salvar_csv(pasta / 'experimentos_brutos.csv',linhas)
    resumo = []
    for l in TAMANHOS:
        rr = [r for r in linhas if r['L']==l]
        medias = lambda k: float(np.mean([float(r[k]) for r in rr if r[k] != ''])) if any(r[k]!='' for r in rr) else ''
        resumo.append(dict(L=l, N=n, custo_medio=medias('custo_otimo'),
                           bfs_tempo_medio_ms=1000*medias('tempo_bfs_s') if l<=limite_bfs else '',
                           bfs_expandidos_medios=medias('expandidos_bfs'),
                           gulosa_tempo_medio_ms=1000*medias('tempo_gulosa_s') if l<=limite_bfs else '',
                           gulosa_expandidos_medios=medias('expandidos_gulosa'),
                           otimizado_tempo_medio_ms=1000*medias('tempo_otimizado_s'),
                           otimizado_expandidos_medios=medias('expandidos_otimizado')))
    salvar_csv(pasta / 'resumo.csv',resumo)
    xs = [r['L'] for r in resumo]
    fig,ax = plt.subplots(figsize=(7.5,4.2))
    ax.plot(xs,[r['custo_medio'] for r in resumo], marker='o',label='BFS/Gulosa: custo ótimo = h(s)')
    ax.set_xscale('log',base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Comprimento L (escala log2)')
    ax.set_ylabel('Custo médio / operações')
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(pasta/'grafico_custo.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.2))
    ax.plot(xs,[r['otimizado_tempo_medio_ms'] for r in resumo],marker='o',label='Gulosa otimizada O(L)')
    small=[r for r in resumo if r['L']<=limite_bfs]
    ax.plot([r['L'] for r in small],[r['bfs_tempo_medio_ms'] for r in small],marker='s',label='BFS convencional')
    ax.plot([r['L'] for r in small],[r['gulosa_tempo_medio_ms'] for r in small],marker='^',label='Gulosa Best-First explícita')
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set_xlabel('Comprimento L (escala log2)')
    ax.set_ylabel('Tempo médio de execução (ms, escala log)')
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(pasta/'grafico_tempo.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.2))
    ax.plot([r['L'] for r in small],[r['bfs_expandidos_medios'] for r in small],marker='s',label='BFS convencional')
    ax.plot([r['L'] for r in small],[r['gulosa_expandidos_medios'] for r in small],marker='^',label='Gulosa Best-First explícita')
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set_xlabel('Comprimento L (escala log2)');ax.set_ylabel('Estados expandidos (média, log)')
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout();fig.savefig(pasta/'grafico_expandidos.png',dpi=180);plt.close(fig)

    s, y = EXEMPLO_INICIAL,EXEMPLO_FINAL
    x0=np.array([int(c) for c in s],dtype=float)
    yt=np.array([int(c) for c in y],dtype=float)
    gd_rows=[]
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    for lam in (0., .05, .25, 1., 4.):
        gd=gradiente_descendente(s,y,lam)
        bits=bits_discretizados(gd.x)
        z=peso_flip(gd.x,x0)
        gd_rows.append(dict(lambda_valor=lam,eta=gd.eta,iteracoes=gd.iteracoes,
                            erro=erro(gd.x,yt),custo_adjacencia=custo_adjacencia(z),
                            objetivo=gd.historico[-1],
                            bits_discretizados=bits,
                            bits_errados=sum(a!=b for a,b in zip(bits,y)),
                            blocos_bits=len(blocos_um(diferencas(bits,s)))))
        lacuna = np.maximum(np.asarray(gd.historico) - gd.historico[-1], 1e-15)
        ax.plot(lacuna,label=f'λ={lam:g}')
    ax.set_yscale('log')
    ax.set_xlabel('Iteração');ax.set_ylabel('J(x) - J(final) (escala log)');ax.grid(alpha=.3)
    ax.legend();fig.tight_layout();fig.savefig(pasta/'grafico_gradiente.png',dpi=180);plt.close(fig)
    salvar_csv(pasta/'gradiente_lambda.csv',gd_rows)
    fig,ax=plt.subplots(figsize=(7.5,3.6))
    for lam in (0., .25, 1., 4.):
        gd=gradiente_descendente(s,y,lam)
        ax.plot(range(1,len(s)+1),peso_flip(gd.x,x0),marker='.',label=f'λ={lam:g}')
    ax.set_ylim(-.03,1.05);ax.set_xlabel('Posição i (1-based)');ax.set_ylabel('Peso contínuo de flip z_i')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(pasta/'grafico_pesos_flip.png',dpi=180);plt.close(fig)
    print('\nResumo salvo em',pasta/'resumo.csv')
    for r in gd_rows: print('GD',r)


def testes() -> None:
    # Operações elementares.
    assert flip_bit('0000',2)=='0100'
    assert flip_intervalo('0101',2,4)=='0010'
    assert blocos_um('001011100')==[(3,3),(5,7)]
    assert heuristica(EXEMPLO_INICIAL,EXEMPLO_FINAL)==5
    assert aplicar_caminho(EXEMPLO_INICIAL,greedy_otimizado(EXEMPLO_INICIAL,EXEMPLO_FINAL).caminho)==EXEMPLO_FINAL
    # Comparação exaustiva de custo ótimo nos 256 pares de L=4.
    for i in range(16):
        for j in range(16):
            a,b=f'{i:04b}',f'{j:04b}'
            assert bfs(a,b).custo==greedy_best_first(a,b).custo==greedy_otimizado(a,b).custo
    # Gradiente analítico versus derivada numérica.
    rng=np.random.default_rng(12)
    x0=np.array([0.,1.,1.,0.,1.,0.]);y=np.array([1.,1.,0.,0.,0.,1.]);x=rng.uniform(.1,.9,len(x0))
    lam=.7;eps=1e-6;g=gradiente(x,x0,y,lam)
    for k in range(len(x)):
        dx=np.zeros_like(x);dx[k]=eps
        numerical=(objetivo(x+dx,x0,y,lam)-objetivo(x-dx,x0,y,lam))/(2*eps)
        assert abs(g[k]-numerical)<1e-7,(k,g[k],numerical)
    gd=gradiente_descendente('100','000',0.)
    assert erro(gd.x,np.zeros(3))<1e-14
    for lam in (0,.05,.25,1,4):
        gd=gradiente_descendente(EXEMPLO_INICIAL,EXEMPLO_FINAL,lam)
        assert gd.historico[-1]<=gd.historico[0]+1e-12
        assert all(x>=-1e-12 and x<=1+1e-12 for x in gd.x)
        assert np.all(np.diff(gd.historico)<=1e-8)
    print('Todos os testes passaram (256 pares de L=4 + gradiente numérico).')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true',help='Exemplo do enunciado e GD')
    parser.add_argument('--experimentos',action='store_true',help='Gera CSVs e figuras')
    parser.add_argument('--testes',action='store_true',help='Executa testes de correção')
    parser.add_argument('--todos',action='store_true',help='Executa testes, demonstração e experimentos')
    parser.add_argument('--n',type=int,default=30,help='Número de pares por tamanho (padrão: 30)')
    parser.add_argument('--seed',type=int,default=2026,help='Semente pseudoaleatória (padrão: 2026)')
    parser.add_argument('--saida',type=str,default='resultados',help='Pasta de saída dos experimentos')
    args=parser.parse_args()
    if args.todos or args.testes or not (args.demo or args.experimentos):testes()
    if args.todos or args.demo:demo()
    if args.todos or args.experimentos:experimentos(args.n,args.seed,args.saida)
