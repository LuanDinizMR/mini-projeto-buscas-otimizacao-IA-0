from __future__ import annotations

"""Demonstrações do exemplo original e do efeito da taxa de aprendizado."""

import numpy as np

from .config import EXEMPLO_INICIAL, EXEMPLO_FINAL
from .operacoes import flip_intervalo
from .buscas import greedy_otimizado
from .gradiente import (
    gradiente_descendente, peso_flip, erro, custo_adjacencia, bits_discretizados
)


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
