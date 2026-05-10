
"""Wyświetlanie planszy szachowej i wizualizacja partii z pliku historii ruchów.

Plik może być dopisywany w czasie rzeczywistym (np. przez moduł wykrywający
ruchy). Funkcja `show_game_from_move_history` odtwarza wszystkie zapisane
ruchy, a następnie monitoruje plik i stosuje nowe ruchy w miarę ich pojawiania
się.

Zawiera też prostą funkcję `show_initial_board()` do wyświetlenia pozycji
startowej.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import font
from typing import List

import chess


# mapowanie symboli figur do znaków Unicode
UNICODE = {
	'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
	'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
}


def initial_board() -> List[List[str]]:
	"""Zwraca 8x8 listę z symbolami figur w stanie początkowym.
	"""
	W = {k: UNICODE[k] for k in ['K', 'Q', 'R', 'B', 'N', 'P']}
	B = {k: UNICODE[k.lower()] for k in ['K', 'Q', 'R', 'B', 'N', 'P']}
	return [
		[B['R'], B['N'], B['B'], B['Q'], B['K'], B['B'], B['N'], B['R']],
		[B['P']] * 8,
		[''] * 8,
		[''] * 8,
		[''] * 8,
		[''] * 8,
		[W['P']] * 8,
		[W['R'], W['N'], W['B'], W['Q'], W['K'], W['B'], W['N'], W['R']],
	]


def board_to_matrix(b: chess.Board) -> List[List[str]]:
	"""Konwertuje `chess.Board()` do 8x8 macierzy z Unicode znakami.

	Macierz ma indeks [0] = ranga 8, [7] = ranga 1 (tak, aby rysować z góry na dół).
	"""
	mat: List[List[str]] = []
	for rank in range(7, -1, -1):  # 7..0 -> rangi 8..1
		row: List[str] = []
		for file in range(0, 8):
			sq = chess.square(file, rank)
			p = b.piece_at(sq)
			if p is None:
				row.append('')
			else:
				row.append(UNICODE.get(p.symbol(), ''))
		mat.append(row)
	return mat


def _draw_matrix_on_canvas(canvas: tk.Canvas, matrix: List[List[str]], square_size: int, piece_font: font.Font):
	canvas.delete('all')
	rows, cols = 8, 8
	light = '#F0D9B5'
	dark = '#B58863'
	width = cols * square_size
	height = rows * square_size

	for r in range(rows):
		for c in range(cols):
			x1 = c * square_size
			y1 = r * square_size
			x2 = x1 + square_size
			y2 = y1 + square_size
			color = light if (r + c) % 2 == 0 else dark
			canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
			piece = matrix[r][c]
			if piece:
				canvas.create_text(
					x1 + square_size / 2,
					y1 + square_size / 2,
					text=piece,
					font=piece_font,
				)

	# osie
	label_font = font.Font(size=max(8, int(square_size * 0.12)))
	for c in range(cols):
		file_char = chr(ord('a') + c)
		canvas.create_text(
			c * square_size + square_size * 0.08,
			height - square_size * 0.08,
			anchor='sw',
			text=file_char,
			font=label_font,
		)
	for r in range(rows):
		rank_char = str(8 - r)
		canvas.create_text(
			square_size * 0.08,
			r * square_size + square_size * 0.08,
			anchor='nw',
			text=rank_char,
			font=label_font,
		)


def show_initial_board(square_size: int = 128) -> None:
	"""Prosty widok pozycji startowej (zachowany dla kompatybilności).
	"""
	board = initial_board()
	cols = 8
	rows = 8
	width = cols * square_size
	height = rows * square_size
	root = tk.Tk()
	root.title("Plansza - stan początkowy")
	canvas = tk.Canvas(root, width=width, height=height)
	canvas.pack()
	piece_font = font.Font(family='Segoe UI Symbol', size=max(10, int(square_size * 0.6)))
	_draw_matrix_on_canvas(canvas, board, square_size, piece_font)
	root.resizable(False, False)
	root.mainloop()


def show_game_from_move_history(file_path: str, square_size: int = 128, replay_delay: int = 300, poll_ms: int = 1000) -> None:
	"""Odtwarza i monitoruje `file_path` z historią ruchów.

	Parametry:
	- file_path: ścieżka do pliku move_history.txt
	- replay_delay: ms pomiędzy odtwarzanymi historycznymi ruchami
	- poll_ms: ms co ile sprawdzać, czy pojawiły się nowe linie
	"""
	root = tk.Tk()
	root.title(f"Wizualizacja partii — {os.path.basename(file_path)}")
	cols = 8
	rows = 8
	width = cols * square_size
	height = rows * square_size
	canvas = tk.Canvas(root, width=width, height=height)
	canvas.pack()
	piece_font = font.Font(family='Segoe UI Symbol', size=max(10, int(square_size * 0.6)))

	board = chess.Board()
	matrix = board_to_matrix(board)
	_draw_matrix_on_canvas(canvas, matrix, square_size, piece_font)

	# pomocnicze: lista ruchów do odtworzenia (UCI), oraz plik otwarty do tailowania
	pending_moves: List[str] = []

	def parse_uci_from_line(line: str) -> str | None:
		# szukamy tokenu po słowie 'APPLIED'
		if 'APPLIED' in line:
			parts = line.split()
			try:
				idx = parts.index('APPLIED')
				if idx + 1 < len(parts):
					return parts[idx + 1]
			except ValueError:
				return None
		return None

	def apply_uci(uci: str) -> bool:
		nonlocal board
		try:
			mv = chess.Move.from_uci(uci)
		except Exception:
			return False
		if mv in board.legal_moves:
			board.push(mv)
			return True
		# Jeżeli ruch nie legalny, spróbuj i tak go zastosować (rzadkie przypadki)
		try:
			board.push(mv)
			return True
		except Exception:
			return False

	def draw_board_from_boardobj():
		mat = board_to_matrix(board)
		_draw_matrix_on_canvas(canvas, mat, square_size, piece_font)

	# Wczytaj wszystkie istniejące linie i przygotuj listę do odtworzenia
	try:
		f = open(file_path, 'r', encoding='utf-8')
	except FileNotFoundError:
		tk.messagebox.showerror('Plik nie znaleziony', f'Nie znaleziono pliku: {file_path}')
		root.destroy()
		return

	all_lines = f.readlines()
	for ln in all_lines:
		u = parse_uci_from_line(ln)
		if u:
			pending_moves.append(u)

	# ustaw wskaźnik na koniec pliku żeby monitorować dopisywane linie
	f.seek(0, os.SEEK_END)

	# Odtywarzanie historycznych ruchów sekwencyjnie
	def replay_step():
		if pending_moves:
			uci = pending_moves.pop(0)
			applied = apply_uci(uci)
			if applied:
				draw_board_from_boardobj()
			# kontynuuj po opóźnieniu
			root.after(replay_delay, replay_step)
		else:
			# po odtworzeniu wszystkich historycznych ruchów, zaczynamy tailować plik
			root.after(poll_ms, poll_file)

	def poll_file():
		# czytaj nowe linie i stosuj je
		lines = f.readlines()
		new_any = False
		for ln in lines:
			u = parse_uci_from_line(ln)
			if u:
				applied = apply_uci(u)
				if applied:
					draw_board_from_boardobj()
				new_any = True
		# jeżeli nic nowego, planuj kolejne sprawdzenie
		root.after(poll_ms, poll_file)

	# start replayu
	root.after(100, replay_step)
	root.resizable(False, False)
	root.mainloop()


if __name__ == '__main__':
	# Domyślnie: pokaż planszę startową. Aby oglądać historię, odpal:
	# from Logika_ruchu.plansza import show_game_from_move_history
	# show_game_from_move_history('ścieżka/do/move_history.txt')
	show_initial_board()

