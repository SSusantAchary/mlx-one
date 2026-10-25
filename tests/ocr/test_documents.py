import threading
import time
from concurrent.futures import CancelledError

import pytest
from PIL import Image

from mlx_one.ocr.documents import PixelTransform, iter_pages, parse_page_range
from mlx_one.ocr.postprocessing import TableParser, florence_regions
from mlx_one.ocr.schemas import OCRInput, OCRLimits


def make_pdf(tmp_path, count=3):
    pytest.importorskip("pypdfium2")
    pages = [Image.new("RGB", (32, 24), "white") for _ in range(count)]
    path = tmp_path / "pages.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:])
    for page in pages:
        page.close()
    return path


def test_coordinates_inverse_and_crop():
    transform = PixelTransform((100, 50, 500, 250), (200, 100), (10, 5), (180, 90))
    for point in [(100, 50), (200, 100), (500, 250)]:
        assert transform.to_original(*transform.to_model(*point)) == pytest.approx(point)
    elements = florence_regions(
        "line<loc_0><loc_0><loc_999><loc_0><loc_999><loc_999><loc_0><loc_999>",
        PixelTransform((100, 50, 500, 250), (200, 100)),
        0,
    )
    assert elements[0].bbox == pytest.approx([100.2, 50.1, 499.8, 249.9])
    assert elements[0].confidence is None
    with pytest.raises(ValueError):
        florence_regions("broken<loc_1000>", transform, 0)


def test_page_ranges():
    assert parse_page_range("1-3,5") == (0, 1, 2, 4)
    for value in ["0", "2-1", "a", "1,1", "1-99999999"]:
        with pytest.raises(ValueError):
            parse_page_range(value)


def test_pdf_bounded_selection_cleanup(tmp_path):
    path = make_pdf(tmp_path)
    iterator = iter_pages(OCRInput(path, task="ocr.document", pages=(2, 0), dpi=72))
    page = next(iterator)
    assert page.index == 0
    borrowed = page.image
    page = next(iterator)
    assert page.index == 2
    with pytest.raises(ValueError):
        borrowed.getpixel((0, 0))
    iterator.close()
    with pytest.raises(ValueError):
        page.image.getpixel((0, 0))
    with pytest.raises(ValueError, match="max_pages"):
        list(iter_pages(OCRInput(path, limits=OCRLimits(max_pages=2))))
    with pytest.raises(ValueError, match="outside"):
        list(iter_pages(OCRInput(path, pages=(3,))))


def test_pdf_embedded_policy_no_duplicate_visual(tmp_path):
    path = make_pdf(tmp_path, 1)
    # This PDF is scanned and has no text; embedded_text is explicitly empty,
    # while auto selects visual exactly once.
    pages = iter_pages(OCRInput(path, pdf_policy="embedded_text"))
    page = next(pages)
    assert page.image is None and page.embedded_text == ""
    pages.close()
    pages = iter_pages(OCRInput(path, pdf_policy="auto"))
    page = next(pages)
    assert page.image is not None and page.metadata["path"] == "visual"
    pages.close()


def test_tiff_exif_rgba_and_invalid_images(tmp_path):
    path = tmp_path / "pages.tiff"
    with Image.new("L", (12, 8), "white") as a, Image.new("RGB", (12, 8), "red") as b:
        a.save(path, save_all=True, append_images=[b])
    pages = iter_pages(OCRInput(path))
    assert [p.index for p in pages] == [0, 1]
    with Image.new("RGBA", (8, 4), (0, 0, 0, 0)) as image:
        pages = iter_pages(OCRInput(image))
        page = next(pages)
        assert page.image.getpixel((0, 0)) == (255, 255, 255)
        pages.close()
    with pytest.raises(ValueError):
        list(iter_pages(OCRInput("https://invalid/doc.png")))
    with pytest.raises(ValueError):
        list(iter_pages(OCRInput(b"x" * 20, limits=OCRLimits(max_file_bytes=10))))
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        list(iter_pages(OCRInput(path), cancel=event))
    with pytest.raises(TimeoutError):
        list(iter_pages(OCRInput(path), deadline=time.monotonic() - 1))


def test_table_parser_preserves_cells_and_spans():
    parser = TableParser()
    parser.feed('<table><tr><th colspan="2">Name</th></tr><tr><td>A</td><td>B</td></tr></table>')
    rows = parser.tables[0]["rows"]
    assert rows[0][0]["colspan"] == 2 and rows[0][0]["header"]
    assert rows[1][1]["text"] == "B"
    with pytest.raises(ValueError):
        TableParser().feed('<table><tr><td rowspan="-1">x')
