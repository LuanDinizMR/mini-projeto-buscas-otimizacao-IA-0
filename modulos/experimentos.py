from __future__ import annotations

"""Geração de amostras, comparação de algoritmos, tabelas e gráficos."""

import csv
from pathlib import Path

import numpy as np

from .config import TAMANHOS, EXEMPLO_INICIAL, EXEMPLO_FINAL
from .operacoes import aplicar_caminho, diferencas, blocos_um
from .buscas import heuristica, bfs, greedy_best_first, greedy_otimizado
from .gradiente import (
    gradiente_descendente, peso_flip, erro, custo_adjacencia, bits_discretizados
)


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
