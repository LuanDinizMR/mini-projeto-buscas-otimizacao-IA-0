from __future__ import annotations

"""Busca em largura, busca gulosa Best-First e variante gulosa otimizada."""

import heapq
import time
from collections import deque
from dataclasses import dataclass

from .operacoes import validar, diferencas, blocos_um


@dataclass
class ResultadoBusca:
    caminho: list[tuple[int, int]]  # intervalos inclusivos, índices 1-based
    custo: int
    expandidos: int
    gerados: int
    segundos: float


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
