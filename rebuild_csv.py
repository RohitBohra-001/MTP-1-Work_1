import json
import pandas as pd

records = []
with open('/home/krudra_mtp/rohit/tsv/results/all_experiments.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
df.to_csv('/home/krudra_mtp/rohit/tsv/results/summary_table.csv', index=False)
print(f"Rebuilt summary_table.csv with {len(df)} records.")

