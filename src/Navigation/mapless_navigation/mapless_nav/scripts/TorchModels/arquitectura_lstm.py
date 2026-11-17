import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_final_architecture():
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)

    # Paleta de colores profesional
    colors = {
        'lom': '#9ecae1',       # Azul
        'target': '#fdae6b',    # Naranja
        'cnn': '#a1d99b',       # Verde claro
        'concat': '#ffffb3',    # Amarillo pálido
        'lstm': '#80cdc1',      # Verde azulado
        'mlp': '#e0e0e0',       # Gris
        'out': '#ffff00'        # Amarillo brillante
    }

    def draw_box(x, y, w, h, text, color, fontsize=11):
        # Caja con bordes redondeados
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                      edgecolor="black", facecolor=color, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        # Texto centrado
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', zorder=3)

    def draw_arrow(x1, y1, x2, y2, color='black'):
        # Flechas gruesas y limpias
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', 
                                        mutation_scale=20, linewidth=2, color=color, zorder=1)
        ax.add_patch(arrow)

    # --- 1. TARGET VECTOR (Arriba) ---
    draw_box(1, 7, 3, 1.8, "Target: (d, \u03B8)\nVector 2 x C=3", colors['target'])
    draw_box(5, 7, 2, 1.8, "Target\nFlatten", colors['target'])

    # --- 2. LOM INPUT (Abajo) ---
    # Dibujar 3 capas apiladas para representar C=3
    for i in range(3):
        offset = i * 0.2
        rect = patches.Rectangle((1 - offset, 1.5 + offset), 3, 3, 
                                 edgecolor='black', facecolor=colors['lom'], linewidth=1.5, zorder=2)
        ax.add_patch(rect)
    # Texto de LOM
    ax.text(2.5, 1.0, "LOM Input\n(Stacked C=3)", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    
    # --- 3. CNN BLOCK ---
    draw_box(5, 2, 2.5, 3, "Conv (ReLU)\nMaxPool\nFlatten", colors['cnn'])

    # --- 4. CONCATENATION ---
    draw_box(8.5, 3.5, 2, 3, "Concat\nFeatures\n+ Target", colors['concat'])

    # --- 5. LSTM CELL ---
    draw_box(11.5, 3, 3, 4, "LSTM Cell\n(Stateful)", colors['lstm'])

    # --- 6. MLP (Dense Layers) ---
    draw_box(15.5, 4, 1.2, 2, "Linear\n(ReLU)\n120", colors['mlp'])
    draw_box(17, 4, 1.2, 2, "Linear\n(ReLU)\n80", colors['mlp'])
    draw_box(18.5, 4, 1.2, 2, "Linear\n(Tanh)\n(vx,vy,wz)", colors['out'])

    # --- CONEXIONES (Flechas) ---
    # Target -> Flatten
    draw_arrow(4.0, 7.9, 5.0, 7.9)
    # Flatten -> Concat
    draw_arrow(7.0, 7.0, 8.5, 6.0)
    
    # LOM -> CNN
    draw_arrow(4.0, 3.0, 5.0, 3.0)
    # CNN -> Concat
    draw_arrow(7.5, 3.5, 8.5, 4.0)
    
    # Concat -> LSTM
    draw_arrow(10.5, 5.0, 11.5, 5.0)
    
    # LSTM -> MLP
    draw_arrow(14.5, 5.0, 15.5, 5.0)
    # MLP -> MLP
    draw_arrow(16.7, 5.0, 17.0, 5.0)
    # MLP -> Output
    draw_arrow(18.2, 5.0, 18.5, 5.0)

    # --- RECURRENCE LOOP (Flujo de estado) ---
    # Salida por abajo
    ax.plot([13.0, 13.0], [3.0, 1.5], color='red', linewidth=2.5, zorder=1)
    # Hacia la izquierda
    ax.plot([13.0, 11.0], [1.5, 1.5], color='red', linewidth=2.5, zorder=1)
    # Subida hacia el LSTM
    ax.plot([11.0, 11.0], [1.5, 4.0], color='red', linewidth=2.5, zorder=1)
    
    # Punta de flecha final
    arrow = patches.FancyArrowPatch((11.0, 4.0), (11.5, 4.0), arrowstyle='-|>', 
                                    mutation_scale=20, linewidth=2.5, color='red', zorder=3)
    ax.add_patch(arrow)
    
    # Etiqueta de la recurrencia
    ax.text(12.0, 1.0, "(h_t, c_t) \u2192 next frame", color='red', 
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Título general
    plt.title("LSTM Stateful Architecture for Mapless Navigation (C=3)", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig("arquitectura_lstm_final.png", bbox_inches='tight', dpi=300)
    print("Imagen guardada como 'arquitectura_lstm_final.png'")
    plt.show()

if __name__ == "__main__":
    draw_final_architecture()