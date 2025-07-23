import random

d = {}  # 示例字典（可能为空）
keys = list(d.keys())
random_key = random.choice(keys) if keys else None

print(random_key)
print(random_key == None)