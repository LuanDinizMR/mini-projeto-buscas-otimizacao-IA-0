#!/usr/bin/env python3
"""Ponto de entrada do mini-projeto de buscas clássicas e otimização.

Exemplos:
    python main.py --testes
    python main.py --demo
    python main.py --experimentos --n 30 --seed 2026 --saida resultados
    python main.py --todos

"""
from __future__ import annotations

import argparse

from modulos.demonstracao import demo
from modulos.experimentos import experimentos
from modulos.testes import testes


def main(argv: list[str] | None = None) -> None:
    """Interpreta os argumentos e executa os módulos solicitados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Exemplo do enunciado e GD')
    parser.add_argument('--experimentos', action='store_true', help='Gera CSVs e figuras')
    parser.add_argument('--testes', action='store_true', help='Executa testes de correção')
    parser.add_argument('--todos', action='store_true', help='Executa testes, demonstração e experimentos')
    parser.add_argument('--n', type=int, default=30, help='Número de pares por tamanho (padrão: 30)')
    parser.add_argument('--seed', type=int, default=2026, help='Semente pseudoaleatória (padrão: 2026)')
    parser.add_argument('--saida', type=str, default='resultados', help='Pasta de saída dos experimentos')
    args = parser.parse_args(argv)

    if args.todos or args.testes or not (args.demo or args.experimentos):
        testes()
    if args.todos or args.demo:
        demo()
    if args.todos or args.experimentos:
        experimentos(n=args.n, seed=args.seed, saida=args.saida)


if __name__ == '__main__':
    main()
