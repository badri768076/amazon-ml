import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import polars as pl

raw_dir = Path("artifacts/raw")
s1 = pl.read_parquet(raw_dir / "test_source1.parquet")
s2 = pl.read_parquet(raw_dir / "test_source2.parquet")
s3 = pl.read_parquet(raw_dir / "test_source3.parquet")
m_path = Path("../../output/matching_results.tsv")
if not m_path.exists():
    m_path = Path("output/matching_results.tsv")

matches = pl.read_csv(m_path, separator="\t")
joined = matches.filter(pl.col("matched_entity_ids").str.len_chars() > 0).join(s1, left_on="source1_entity_id", right_on="entity_id")

out_rows = []
for country in ["US", "India", "France"]:
    sub = joined.filter(pl.col("country") == country).head(4)
    for r in sub.iter_rows(named=True):
        mids = r["matched_entity_ids"].split(",")
        for mid in mids[:2]:
            cand_df = s2 if mid.startswith("S2-") else s3
            cr = cand_df.filter(pl.col("entity_id") == mid)
            if cr.height:
                crow = cr.row(0, named=True)
                out_rows.append({
                    "country": country,
                    "s1_id": r["source1_entity_id"],
                    "s1_name": r["business_name"],
                    "s1_address": r["business_address"],
                    "cand_id": mid,
                    "cand_source": "Source 2" if mid.startswith("S2-") else "Source 3",
                    "cand_name": crow["business_name"],
                    "cand_address": crow["business_address"]
                })

df_out = pl.DataFrame(out_rows)
csv_path = Path("sample_test_data.csv")
df_out.write_csv(csv_path)
print(f"Wrote {df_out.height} test pairs to {csv_path.resolve()}")

# Also print the first 6 pairs cleanly
for i, row in enumerate(out_rows[:6]):
    print(f"\n--- [Test Case #{i+1} - {row['country']}] ---")
    print(f"Source 1 ({row['s1_id']}):")
    print(f"  Name   : \"{row['s1_name']}\"")
    print(f"  Address: \"{row['s1_address']}\"")
    print(f"Candidate ({row['cand_id']} - {row['cand_source']}):")
    print(f"  Name   : \"{row['cand_name']}\"")
    print(f"  Address: \"{row['cand_address']}\"")
