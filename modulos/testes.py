from __future__ import annotations

"""Testes de regressão e validação numérica dos algoritmos."""

import numpy as np

from .config import EXEMPLO_INICIAL, EXEMPLO_FINAL
from .operacoes import flip_bit, flip_intervalo, blocos_um, aplicar_caminho
from .buscas import heuristica, bfs, greedy_best_first, greedy_otimizado
from .gradiente import gradiente, objetivo, gradiente_descendente, erro


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
