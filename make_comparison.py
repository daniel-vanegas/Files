import os
import glob
import random
import sys
from PIL import Image, ImageDraw

base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
lom_dir = os.path.join(base_dir, "lom")
tlom_dir = os.path.join(base_dir, "tlom")

lom_files = set(os.path.basename(f) for f in glob.glob(os.path.join(lom_dir, "*.png")))
tlom_files = set(os.path.basename(f) for f in glob.glob(os.path.join(tlom_dir, "*.png")))
common_files = sorted(list(lom_files.intersection(tlom_files)))

if len(common_files) < 10:
    print(f"Error: Se necesitan al menos 10 frames en comun en {base_dir}")
    exit(1)

# Seleccionar 10 frames aleatorios
random.seed(42)
selected_files = random.sample(common_files, 10)

sample_img = Image.open(os.path.join(lom_dir, selected_files[0]))
w, h = sample_img.size

# Escalar x2
scale = 2
w_s, h_s = w * scale, h * scale

# Margen de separacion entre imagenes (en pixeles)
margin = 10

# Calcular el tamaño del lienzo (10 columnas con margenes)
canvas_w = (w_s + margin) * 10 + margin
canvas_h = h_s * 2 + 40

# Fondo gris oscuro
canvas = Image.new('RGB', (canvas_w, canvas_h), color=(50, 50, 50))
draw = ImageDraw.Draw(canvas)

for col, fname in enumerate(selected_files):
    frame_num = fname.replace(".png", "").lstrip("0") or "0"
    
    img_lom = Image.open(os.path.join(lom_dir, fname)).convert('L').resize((w_s, h_s), Image.NEAREST).convert('RGB')
    img_tlom = Image.open(os.path.join(tlom_dir, fname)).convert('L').resize((w_s, h_s), Image.NEAREST).convert('RGB')
    
    # Calcular el desplazamiento X con el margen
    x_offset = margin + col * (w_s + margin)
    
    # Fila superior: LOM
    canvas.paste(img_lom, (x_offset, 0))
    draw.text((x_offset + 5, 5), f"F{frame_num} LOM", fill=(255, 0, 0))
    
    # Fila inferior: T-LOM
    canvas.paste(img_tlom, (x_offset, h_s + 20))
    draw.text((x_offset + 5, h_s + 25), f"F{frame_num} T-LOM", fill=(0, 255, 0))

output_path = os.path.join(base_dir, "comparativa.png")
canvas.save(output_path)
print(f"Exito! Imagen guardada en: {output_path}")
