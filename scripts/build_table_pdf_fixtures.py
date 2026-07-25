"""Build synthetic but table-layout PDFs for live document QA (PyMuPDF only)."""

from __future__ import annotations

from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "stress"


def _draw_table(
    page: fitz.Page,
    *,
    origin: tuple[float, float],
    col_widths: list[float],
    rows: list[list[str]],
    fontsize: float = 9.5,
    fontname: str = "helv",
) -> None:
    x0, y0 = origin
    row_h = fontsize + 12
    for r_i, row in enumerate(rows):
        x = x0
        y = y0 + r_i * row_h
        for c_i, cell in enumerate(row):
            w = col_widths[c_i]
            rect = fitz.Rect(x, y, x + w, y + row_h)
            page.draw_rect(rect, color=(0.15, 0.15, 0.15), width=0.6)
            page.insert_textbox(
                fitz.Rect(x + 3, y + 2, x + w - 2, y + row_h - 1),
                cell,
                fontsize=fontsize,
                fontname=fontname,
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_LEFT if c_i == 0 else fitz.TEXT_ALIGN_RIGHT,
            )
            x += w


def build_apple_msft_zh_table_pdf(path: Path) -> Path:
    """Bilingual/Chinese-labeled peer table for ZH phrasing matrix."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    font = "china-s"
    page.insert_text(
        (48, 48),
        "同业基本面对比表（合成尽调材料）",
        fontsize=12,
        fontname=font,
    )
    page.insert_text(
        (48, 68),
        "期间：FY2025 | 单位：十亿美元 | Apple / Microsoft",
        fontsize=9,
        fontname=font,
    )
    rows = [
        ["指标", "苹果", "微软"],
        ["营收", "383.3", "245.1"],
        ["EBITDA", "130.1", "128.4"],
        ["营业利润", "118.2", "109.4"],
        ["研发", "31.4", "29.5"],
    ]
    _draw_table(
        page,
        origin=(48, 96),
        col_widths=[140, 120, 120],
        rows=rows,
        fontsize=10,
        fontname=font,
    )
    page.insert_text(
        (48, 260),
        "附注：苹果供应链仍有组装集中度风险；微软云与办公软件为主要增长引擎。\n"
        "本文件为 LumenFin 中文表格解析与话术矩阵测试用合成 PDF。",
        fontsize=9,
        fontname=font,
    )
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((48, 48), "附录：句式校验", fontsize=12, fontname=font)
    page2.insert_text(
        (48, 72),
        "苹果 FY2025 营收为 383.3 billion USD，EBITDA 为 130.1 billion USD，研发为 31.4 billion USD。\n"
        "微软 FY2025 营收为 245.1 billion USD，EBITDA 为 128.4 billion USD，研发为 29.5 billion USD。",
        fontsize=10,
        fontname=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def build_apple_msft_table_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (48, 48),
        "Consolidated Peer Fundamentals Table (Synthetic Diligence Pack)",
        fontsize=12,
        fontname="helv",
    )
    page.insert_text(
        (48, 68),
        "Period: FY2025 | Units: USD billions unless noted | Source: internal model pack",
        fontsize=9,
        fontname="helv",
    )
    rows = [
        ["Metric", "Apple", "Microsoft"],
        ["Revenue", "383.3", "245.1"],
        ["EBITDA", "130.1", "128.4"],
        ["Operating Income", "118.2", "109.4"],
        ["R&D Expense", "31.4", "29.5"],
        ["Cash & Equivalents", "65.2", "78.0"],
        ["Total Debt", "106.6", "67.1"],
    ]
    _draw_table(
        page,
        origin=(48, 96),
        col_widths=[160, 120, 120],
        rows=rows,
        fontsize=10,
    )
    page.insert_text(
        (48, 280),
        "Narrative notes",
        fontsize=11,
        fontname="helv",
    )
    notes = (
        "Apple supply chain risk remains medium due to assembly concentration in Greater China.\n"
        "Microsoft Azure and Office remain primary growth engines; compliance focus includes cloud residency.\n"
        "Management tone: Apple emphasized services mix; Microsoft highlighted AI Copilot monetization.\n"
        "This PDF is synthetic for LumenFin table-extraction and provenance testing."
    )
    page.insert_text((48, 300), notes, fontsize=9, fontname="helv")

    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((48, 48), "Appendix A — Ratio Check Inputs", fontsize=12, fontname="helv")
    page2.insert_text(
        (48, 72),
        "Apple FY2025 Revenue was 383.3 billion USD. EBITDA was 130.1 billion USD.\n"
        "Microsoft FY2025 Revenue was 245.1 billion USD. EBITDA was 128.4 billion USD.\n"
        "Apple R&D was 31.4 billion USD. Microsoft research and development was 29.5 billion USD.",
        fontsize=10,
        fontname="helv",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def build_tsmc_single_table_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (48, 48),
        "TSMC FY2025 Operating Metrics (Synthetic Table PDF)",
        fontsize=12,
        fontname="helv",
    )
    rows = [
        ["Line Item", "FY2025"],
        ["Company", "TSMC"],
        ["Revenue (USD bn)", "88.3"],
        ["EBITDA (USD bn)", "58.1"],
        ["R&D (USD bn)", "7.2"],
        ["Gross Margin %", "54.7"],
    ]
    _draw_table(page, origin=(48, 80), col_widths=[220, 120], rows=rows, fontsize=10)
    page.insert_text(
        (48, 220),
        "Supply chain: advanced packaging capacity and EUV tool lead times remain key risks.\n"
        "Compliance: export-control screening for certain nodes is ongoing.",
        fontsize=9,
        fontname="helv",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def main() -> None:
    a = build_apple_msft_table_pdf(OUT / "apple_msft_fy2025_table.pdf")
    b = build_tsmc_single_table_pdf(OUT / "tsmc_fy2025_table.pdf")
    c = build_apple_msft_zh_table_pdf(OUT / "apple_msft_fy2025_table_zh.pdf")
    print(a)
    print(b)
    print(c)
    # quick extract sanity
    for p in (a, b, c):
        d = fitz.open(p)
        text = "\n".join(pg.get_text("text") for pg in d)
        print("---", p.name, "chars", len(text))
        print(text[:500])
        print()


if __name__ == "__main__":
    main()
