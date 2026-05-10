

Krótki opis architektury modułu `gotowe.py`

## Cel

Z kamery na żywo odczytać układ bierek na szachownicy, porównywać kolejne stany i **zatwierdzać legalne ruchy** na wirtualnej planszy (`python-chess`), z podglądem GUI.

## Przepływ danych

1. **Kalibracja** — użytkownik zaznacza 4 narożniki planszy i 9 linii pionowych oraz 9 poziomych w obrazie po transformacji perspektywy. Ustawienia można zapisać w `camera_calibration.json`.
2. **Przechwyt klatki** — `VideoCapture` → opcjonalnie `warpPerspective` do stałego rozmiaru (`OUTPUT_SIZE`).
3. **Segmentacja** — na podstawie linii siatki wycinane są 64 pola (każde jako obrazek).
4. **Klasyfikacja** — model Keras (`models/model_szachowy.keras`) przypisuje każdemu polu etykietę: `black` / `white` / `empty`.
5. **Stabilizacja** — z kilku kolejnych predykcji budowany jest **stabilny stan** (głosowanie większościowe po polach); przy dużym szumie klatka jest odrzucana.
6. **Detekcja ruchu** — porównanie stabilnego stanu z poprzednim:
   - **DELTA**: jawna zmiana „źródło → cel” (w tym bicie przez wykrycie zmiany koloru na polu docelowym),
   - **LEGAL**: gdy DELTA nie wystarcza, dopasowanie do **legalnego** ruchu minimalizującego odległość Hamminga między oczekiwanym a obserwowanym stanem (z progami `dist` / `margin` i dodatkowym potwierdzeniem przy słabych kandydatach).
7. **Potwierdzenie** — ten sam kandydat musi pojawić się **wiele razy z rzędu** (`REQUIRED_CONSECUTIVE` + adaptacja przy szumie, niskiej pewności modelu, ruchu sceny).
8. **Filtrowanie dłoni** — wysoki odsetek zmienionych pikseli między klatkami resetuje kandydata (unikamy „ruchów” podczas przesuwania figury).
9. **Zapis** — zatwierdzone ruchy trafiają do `move_history.txt`; GUI tkinter pokazuje aktualną pozycję; możliwy cof (`undo` / Ctrl+Z w oknie planszy).

## Główne pliki

| Plik | Rola |
|------|------|
| `gotowe.py` | Cały pipeline: kamera, kalibracja, wątek analizy, GUI |
| `camera_calibration.json` | Zapis kalibracji (opcjonalnie) |
| `move_history.txt` | Log zatwierdzonych ruchów |
| `../models/model_szachowy.keras` | Model klasyfikacji pól |
| `plansza.py` | Rysowanie szachownicy w tkinter |

## Uruchomienie (skrót)

```bash
python gotowe.py
```

Sterowanie w oknie kamery: kalibracja (`v` / `h` linie, `r` reset), `q` — start analizy w tle i okno z planszą. Zamknięcie podglądu OpenCV kończy sesję.

## Zależności (orientacyjnie)

OpenCV, NumPy, TensorFlow/Keras, `python-chess`, tkinter (standardowo z Pythonem na Windows).
