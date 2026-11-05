import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, "../../data"))

classes = ["straight", "semi_complex", "complex", "v_complex"]

print(f"Analizando pools en: {base_dir}\n")
print(f"{'Scenario':<10} | {'Total':<6} | {'Straight':<8} | {'Semi':<6} | {'Complex':<7} | {'V-Complex':<9}")
print("-" * 65)

total_all = {c: 0 for c in classes}
grand_total = 0

if not os.path.exists(base_dir):
    print(f"Error: No se encontro la carpeta {base_dir}")
    exit(1)

for scenario in sorted(os.listdir(base_dir)):
    if not scenario.startswith("E") or not os.path.isdir(os.path.join(base_dir, scenario)):
        continue
    
    pool_file = os.path.join(base_dir, scenario, "path_pool.json")
    if not os.path.exists(pool_file):
        print(f"{scenario:<10} | {'MISSING':<6} |")
        continue
    
    try:
        with open(pool_file, 'r') as f:
            data = json.load(f)
        
        paths = data.get("paths", [])
        counts = {c: 0 for c in classes}
        
        for p in paths:
            cls = p.get("class")
            if cls in counts:
                counts[cls] += 1
                
        total = sum(counts.values())
        grand_total += total
        for c in classes:
            total_all[c] += counts[c]
            
        print(f"{scenario:<10} | {total:<6} | {counts['straight']:<8} | {counts['semi_complex']:<6} | {counts['complex']:<7} | {counts['v_complex']:<9}")
        
    except Exception as e:
        print(f"{scenario:<10} | ERROR: {e}")

print("-" * 65)
print(f"{'TOTAL':<10} | {grand_total:<6} | {total_all['straight']:<8} | {total_all['semi_complex']:<6} | {total_all['complex']:<7} | {total_all['v_complex']:<9}")
