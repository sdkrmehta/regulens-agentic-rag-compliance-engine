import os
import sys
import time
import urllib.request
from pathlib import Path

PDF_SOURCES = {
    "rbi_kyc_master_direction.pdf": [
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/18MDR04012024227EF10313334FE58639234B38A3A401.PDF",
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/18MD27052021B7E0DA0A0BD64C7291B24B2588BEFE02.PDF"
    ],
    "rbi_digital_lending_guidelines.pdf": [
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/GUIDELINESDIGITALLENDINGD5C35A71D8124A0E92AEB940A7D25BB3.PDF",
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/GUIDELINESDIGITALLENDINGD5370AD5D4A24794BCF9D9AE5D8ED439.PDF"
    ],
    "rbi_cyber_resilience_master_direction.pdf": [
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD80071120239A5D577884144414A4E58E9B9735FBD1.PDF",
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD5NOV2026C121735165E044FE9AB227A6CF04FF95.PDF"
    ]
}

def download_file(filename: str, urls: list[str], dest_dir: Path):
    dest_path = dest_dir / filename
    if dest_path.exists() and dest_path.stat().st_size > 10000:
        print(f"[OK] {filename} already exists ({dest_path.stat().st_size} bytes). Skipping.")
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for url in urls:
        print(f"Attempting download for {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                if len(content) > 10000:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    print(f"[SUCCESS] Downloaded {filename} ({len(content)} bytes)")
                    return True
                else:
                    print(f"[WARNING] Downloaded content too small ({len(content)} bytes), trying fallback.")
        except Exception as e:
            print(f"[ERROR] Failed from {url}: {e}")
            time.sleep(1)

    return False

def main():
    dest_dir = Path("data/raw_pdfs")
    dest_dir.mkdir(parents=True, exist_ok=True)
    all_success = True
    for filename, urls in PDF_SOURCES.items():
        success = download_file(filename, urls, dest_dir)
        if not success:
            all_success = False

    if all_success:
        print("\nAll RBI Master Direction PDFs are downloaded and verified in data/raw_pdfs/!")
        sys.exit(0)
    else:
        print("\nSome PDFs could not be downloaded automatically.")
        sys.exit(1)

if __name__ == "__main__":
    main()
