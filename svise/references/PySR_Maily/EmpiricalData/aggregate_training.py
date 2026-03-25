"""
Aggregiert alle results_chunks_*.csv Dateien aus einem Verzeichnis
zu einer einzigen kombinierten CSV-Datei.

Verwendung:
    python aggregate_results.py

Konfiguration:
    INPUT_DIR  - Verzeichnis mit den CSV-Dateien
    OUTPUT_FILE - Pfad zur Ausgabedatei
"""

import pandas as pd
import glob
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_DIR   = "/home/ka/ka_iai/ka_hr7224/PySRCurrent/5minChunks/full_run"
OUTPUT_FILE = "/home/ka/ka_iai/ka_hr7224/PySRCurrent/results_all_combined.csv"
FILE_PATTERN = "results_chunks_*.csv"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    pattern = os.path.join(INPUT_DIR, FILE_PATTERN)
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Keine Dateien gefunden: {pattern}")
        return

    print(f"{len(files)} CSV-Dateien gefunden. Lese ein...")

    dfs = []
    errors = []

    for i, filepath in enumerate(files, 1):
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] gelesen — {len(df)} Zeilen aus {os.path.basename(filepath)}")
        except Exception as e:
            errors.append((filepath, str(e)))
            print(f"  FEHLER bei {os.path.basename(filepath)}: {e}")

    if not dfs:
        print("Keine Daten geladen. Abbruch.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Sortieren nach chunk_id (falls vorhanden)
    if "chunk_id" in combined.columns:
        combined = combined.sort_values("chunk_id").reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✓ Fertig!")
    print(f"  Zeilen gesamt : {len(combined):,}")
    print(f"  Spalten       : {list(combined.columns)}")
    print(f"  Ausgabe       : {OUTPUT_FILE}")

    if errors:
        print(f"\n⚠ {len(errors)} Dateien konnten nicht gelesen werden:")
        for path, msg in errors:
            print(f"  {os.path.basename(path)}: {msg}")


if __name__ == "__main__":
    main()