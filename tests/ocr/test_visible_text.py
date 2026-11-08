"""Plain-text OCR responses preserve recognized values across output formats."""

from types import SimpleNamespace

from mlx_one.ocr.postprocessing import parse_output, visible_text


def test_markdown_table_syntax_is_removed_but_cell_values_are_preserved():
    table = "| Item | Value |\n| :--- | ---: |\n| Sample | -12.50 |\n\nA | B"
    assert visible_text(table) == "Item Value\nSample -12.50\nA | B"


def test_html_cells_and_breaks_do_not_merge_words_or_leave_tags():
    source = "<h1>Invoice</h1><table><tr><td>Amount<br>123.45</td><td>A &amp; B</td></tr></table>"
    result = visible_text(source)
    assert " ".join(result.split()) == "Invoice Amount 123.45 A & B"
    assert visible_text(result) == result


def test_markdown_checkbox_states_numbers_and_literal_underscores_survive():
    result = visible_text("## Form\n- [ ] New\n- [x] Continue\nValue: **12.5**\nName: \\_\\_\\_")
    assert result == "Form\n□ New\n☑ Continue\nValue: 12.5\nName: ___"
    assert visible_text("Value < 12 and x_y") == "Value < 12 and x_y"


def test_markdown_transport_remains_available_alongside_rendered_text():
    bundle = SimpleNamespace(architecture="lightonocr", capabilities=SimpleNamespace(markdown=True))
    request = SimpleNamespace(task="ocr.recognize", output_format="markdown")
    raw = "# Title\n<table><tr><td>42</td></tr></table>"
    text, markdown, *_ = parse_output(bundle, request, raw, None, 0)
    assert markdown == raw and " ".join(text.split()) == "Title 42"
