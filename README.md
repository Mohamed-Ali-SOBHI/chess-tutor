# Chess Screen Assistant

Application locale pour :

- capturer tout l'ecran
- detecter automatiquement le plateau d'echecs dans la capture
- extraire le FEN
- demander un meilleur coup a Stockfish limite a 2000 Elo

Le projet a ete refactore pour separer la vision, l'extraction FEN, le moteur et l'interface.

## Prerequis

- Windows
- Python 3.10+
- un environnement virtuel Python
- Stockfish si tu veux la recommandation de coup

## Installation

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application avec bouton

Le plus simple :

```powershell
.\launch_chess_assistant.bat
```

Ou directement :

```powershell
.\venv\Scripts\python.exe .\chess_assistant.py
```

## Utilisation

1. Ouvre ton ecran avec le plateau visible.
2. Lance l'application.
3. Renseigne `stockfish.exe` si tu veux une recommandation de coup.
4. Choisis le camp au trait : `Blancs` ou `Noirs`.
5. Clique sur `Capturer l'ecran et recommander`.

L'application :

- fait une capture plein ecran
- detecte le plateau dans l'image complete
- extrait le FEN
- calcule un coup avec Stockfish si le moteur est disponible

Si Stockfish n'est pas configure, l'application affiche quand meme le FEN detecte.

## Configurer Stockfish

Tu peux :

- renseigner le chemin dans le champ de l'application
- ou definir la variable d'environnement `STOCKFISH_PATH`

Exemple :

```powershell
$env:STOCKFISH_PATH="C:\chemin\vers\stockfish.exe"
```

## Lancer la conversion en script

Le point d'entree historique existe encore :

```powershell
.\venv\Scripts\python.exe .\convert_fen.py
```

Ca parcourt les images du dossier et ecrit les resultats dans `fen_results.csv`.

## Structure

- [convert_fen.py](C:/Users/moham/OneDrive/Bureau/chess/convert_fen.py) : facade de compatibilite
- [chess_assistant.py](C:/Users/moham/OneDrive/Bureau/chess/chess_assistant.py) : lancement de l'interface desktop
- [chess_fen/vision.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/vision.py) : detection et crop du plateau
- [chess_fen/predictor.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/predictor.py) : choix du meilleur FEN
- [chess_fen/fen_utils.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/fen_utils.py) : utilitaires FEN
- [chess_fen/engine.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/engine.py) : integration Stockfish
- [chess_fen/service.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/service.py) : orchestration du pipeline
- [chess_fen/capture.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/capture.py) : capture plein ecran
- [chess_fen/ui.py](C:/Users/moham/OneDrive/Bureau/chess/chess_fen/ui.py) : interface Tkinter

## Fichiers utiles

- [fen_results.csv](C:/Users/moham/OneDrive/Bureau/chess/fen_results.csv) : sortie CSV
- [fen_overrides.csv](C:/Users/moham/OneDrive/Bureau/chess/fen_overrides.csv) : corrections manuelles de FEN

## Notes

- Le moteur est configure a `2000` Elo par defaut.
- Le plateau peut etre detecte a partir d'une capture plein ecran, pas seulement d'un screenshot deja recadre.
- Si le plateau n'est pas trouve proprement, le systeme peut retomber sur l'image complete.
