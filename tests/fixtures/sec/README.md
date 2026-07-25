# SEC-Derived Test Fixtures

These fixtures are minimized test extracts derived from publicly accessible
EDGAR filings. They are not complete filings, are not official SEC PDFs, and do
not imply SEC affiliation or endorsement.

## Policy

- Complete filings are downloaded only to `.local-fixtures/sec/downloads/`.
- Full downloads are ignored and never required during ordinary unit tests.
- Committed HTML contains only tables/paragraphs needed by parser tests.
- Derived PDFs contain selected or paraphrased content and explicit labels.
- Images, logos, scripts, exhibits and signature pages are omitted.
- `manifest.json` records issuer, CIK, form, accession, filing date, source URL,
  retrieval date and SHA-256 checksums.

## Commands

Fetch full sources into the ignored cache:

```bash
SEC_USER_AGENT="LumenFin/0.1 contact@example.com" \
python scripts/fetch_sec_fixtures.py fetch
```

Build derived PDFs:

```bash
python scripts/fetch_sec_fixtures.py build
```

Verify committed checksums:

```bash
python scripts/fetch_sec_fixtures.py verify
```

The fetch command uses an identifying `SEC_USER_AGENT`, a bounded request rate,
local caching location and checksum verification. Tests do not repeatedly fetch
from EDGAR.
