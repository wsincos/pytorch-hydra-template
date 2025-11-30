import pandas as pd

# 换成你的文件路径
file_path = "/data1/jinyu_wang/projects/transformer-template/data/datasets/opus-100/train/train-00000-of-00001.parquet"

# 读取前 3 行
df = pd.read_parquet(file_path, engine='pyarrow')
print("=== 列名 (Columns) ===")
print(df.columns.tolist())

print("\n=== 前 1 行数据 (Sample) ===")
print(df.iloc[0].to_dict())

data = df.to_dict(orient='records')
print("\n=== 前 3 行数据 (First 3 Rows) ===")
for i, row in enumerate(data[:3]):
    print(f"Row {i + 1}: {row}")
print("\n=== 数据总行数 (Total Rows) ===")