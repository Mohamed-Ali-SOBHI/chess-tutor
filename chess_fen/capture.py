from PIL import ImageGrab


def capture_full_screen():
    try:
        return ImageGrab.grab(all_screens=True).convert("RGB")
    except Exception as error:
        raise RuntimeError(
            "Impossible de capturer l'ecran. Verifiez que la session Windows est active."
        ) from error
