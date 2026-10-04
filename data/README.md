# Data

Put your source documents in `data/raw/` (PDF or TXT). Use text-based PDFs; scanned
PDFs need OCR first.

Suggested public sources (download the official PDFs yourself and check each
site's terms of use):

- **Constitution of India** - India Code (indiacode.nic.in) or the Legislative Department (legislative.gov.in)
- **RBI circulars and master directions** - rbi.org.in
- **Income Tax FAQs / guides** - incometaxindia.gov.in

Tips:
- Use clear file names, e.g. `constitution_of_india.pdf`. The file name becomes the
  document title shown in citations.
- Start with 2 or 3 documents so indexing is quick, then add more.
- After adding or changing files, rebuild the index: `python -m src.ingest`
