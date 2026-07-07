from __future__ import annotations

from pathlib import Path

import fitz

PAGE_1 = """NVIDIA Corporation - FY2025 Earnings Release Excerpt (Sample for LumenFin Demo)

Company: NVIDIA Corporation (NASDAQ: NVDA)
Fiscal Year: FY2025

Financial Highlights
Total revenue was 130.5 billion USD, up 114 percent year over year.
Data Center revenue was 115.2 billion USD, representing approximately 88 percent of total revenue.
Gross margin was 75.0 percent on a GAAP basis.

Profitability
GAAP operating income was 81.5 billion USD.
EBITDA was approximately 75.2 billion USD.
Research and development expenses were 12.8 billion USD, or 9.8 percent of revenue.

Management Commentary
CEO stated: We are seeing sustained demand for AI infrastructure build-out across cloud service providers and enterprises.
CFO noted product roadmap confidence and strong platform adoption for CUDA and Blackwell architecture.

Supply Chain and Risk Disclosures
The company disclosed medium supply chain concentration risk in advanced packaging and HBM memory suppliers.
Geopolitical export controls remain a monitored regulatory risk for data center GPU shipments.
"""

PAGE_2 = """NVIDIA FY2025 - Supplemental Risk Factors (Sample)

Operational risks include dependency on third-party foundry and substrate suppliers in Asia.
Competitive intensity in AI accelerators may pressure pricing over time.

Forward-looking statements in this excerpt are for demonstration only.
This is not an official SEC filing; it is a synthetic diligence document for RAG testing.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "fixtures" / "nvidia_fy2025_earnings_excerpt.pdf"
    path.parent.mkdir(exist_ok=True)
    doc = fitz.open()
    for text in (PAGE_1, PAGE_2):
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()
    print(path)


if __name__ == "__main__":
    main()
