from __future__ import annotations

"""Validação, flips e manipulação de sequências de bits."""

# Representação das operações de inversão em sequências binárias.


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


def aplicar_caminho(inicial: str, caminho: list[tuple[int, int]]) -> str:
    estado = inicial
    for ini, fim in caminho:
        estado = flip_intervalo(estado, ini, fim)
    return estado
