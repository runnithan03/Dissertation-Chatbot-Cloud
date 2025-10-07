import pickle

with open("chunks.pkl", "rb") as f:
    raw = pickle.load(f)

print(type(raw))
if isinstance(raw, dict):
    # show first 2 items
    for k, v in list(raw.items())[:2]:
        print("KEY:", k)
        print("VALUE TYPE:", type(v))
        print("VALUE PREVIEW:", str(v)[:200])
        print()
elif isinstance(raw, list):
    for x in raw[:2]:
        print("ELEM TYPE:", type(x))
        print("PREVIEW:", str(x)[:200])
        print()