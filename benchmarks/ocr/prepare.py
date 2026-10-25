"""Generate fixed synthetic fixtures. No downloaded corpus or handwriting is fabricated."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, __version__

ROOT = Path("docs/benchmarks/ocr/fixtures")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    fixtures = []
    specifications = [
        ("OCR-S1", "line", "Invoice number 12345", (600, 60)),
        ("OCR-S2", "receipt", "SHOP\n2026-10-25\nTotal 42.50", (600, 800)),
        ("OCR-S3", "page", "Document heading\nFirst paragraph.\nSecond paragraph.", (800, 1100)),
        ("OCR-S4", "dense", "Left column text.\nRight column text.\nFootnote one.", (800, 1100)),
        ("OCR-S5", "table", "Name | Quantity\nApple | 3\nPear | 2", (800, 600)),
        ("OCR-S6", "formula", r"E = mc^2", (600, 100)),
        ("OCR-S9", "degraded", "Read this degraded document.", (800, 300)),
    ]
    for id, name, text, size in specifications:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        if id == "OCR-S4":
            draw.text((20, 30), "Left column text.", fill="black", font_size=22)
            draw.text((410, 30), "Right column text.", fill="black", font_size=22)
            draw.text((20, 1000), "Footnote one.", fill="black", font_size=12)
        else:
            draw.multiline_text((20, 15), text, fill="black", font_size=22, spacing=14)
        if id == "OCR-S9":
            changed = image.rotate(3, fillcolor="white").filter(ImageFilter.GaussianBlur(0.8))
            image.close()
            image = changed
        path = ROOT / (id + ".png")
        image.save(path)
        image.close()
        fixtures.append(
            dict(
                id=id,
                name=name,
                path=str(path),
                reference=text,
                kind="synthetic",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    pages = [Image.new("RGB", (400, 550), "white") for _ in range(10)]
    for i, page in enumerate(pages):
        ImageDraw.Draw(page).text((15, 20), f"Page {i + 1}", fill="black", font_size=22)
    path = ROOT / "OCR-S7.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:])
    for page in pages:
        page.close()
    fixtures.append(
        dict(
            id="OCR-S7",
            name="ten-page PDF",
            path=str(path),
            reference="\n\n".join(f"Page {i + 1}" for i in range(10)),
            kind="synthetic",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "generator": "pillow " + __version__,
                "license": "CC0-1.0; generated project fixtures",
                "fixtures": fixtures,
                "missing": [
                    {
                        "id": "OCR-S8",
                        "reason": "Provide licensed real handwriting and transcript; "
                        "printed synthetic text is not a handwriting benchmark.",
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
