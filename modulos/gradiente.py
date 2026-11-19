from __future__ import annotations

"""Otimização contínua, funções de custo e gradiente descendente projetado."""

from dataclasses import dataclass

import numpy as np

from .operacoes import validar


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
