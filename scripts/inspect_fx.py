import io
import pandas as pd
from src.data_collection import create_robust_session, BCB_EVOLUTIVO_BASE

session = create_robust_session()
r = session.get(BCB_EVOLUTIVO_BASE, params={"serie": "tipo_de_cambio", "format": "csv"}, timeout=15)
df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="utf-8-sig")

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head(5).to_string())
print(df.tail(15).to_string())