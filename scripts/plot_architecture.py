"""Generate architecture diagram for FaceFlash."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGS_DIR = Path(__file__).parent.parent / "docs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def draw_pipeline():
    """Main pipeline diagram."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # Boxes
    boxes = [
        (0.5, 1.2, "Image\nInput", "#ecf0f1"),
        (2.3, 1.2, "Face\nDetection", "#3498db"),
        (4.1, 1.2, "ArcFace\nEmbedding", "#3498db"),
        (5.9, 1.2, "PCA Binary\nQuantize", "#e74c3c"),
        (7.7, 1.2, "Hamming\nSearch", "#e74c3c"),
        (9.5, 1.2, "Cosine\nRerank", "#e74c3c"),
        (11.3, 1.2, "Result", "#2ecc71"),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black',
                                        linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i+1][0] - 0.7
        ax.annotate('', xy=(x2, 1.2), xytext=(x1, 1.2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    # Labels below
    labels = [
        (0.5, 0.3, "JPG/PNG"),
        (2.3, 0.3, "OpenCV\n~10ms"),
        (4.1, 0.3, "ONNX 512-d\n~30ms"),
        (5.9, 0.3, "64 bytes\n<1ms"),
        (7.7, 0.3, "Rust POPCNT\n0.3ms @100K"),
        (9.5, 0.3, "Top-K float\n0.04ms"),
        (11.3, 0.3, "Name +\nConfidence"),
    ]
    for x, y, text in labels:
        ax.text(x, y, text, ha='center', va='center', fontsize=7, color='#555')

    # Title
    ax.text(6, 3.1, "FaceFlash Pipeline", ha='center', fontsize=14, fontweight='bold')

    # Legend
    legend_items = [
        mpatches.Patch(color='#3498db', label='Standard (detection + embedding)'),
        mpatches.Patch(color='#e74c3c', label='FaceFlash (quantize + search + rerank)'),
        mpatches.Patch(color='#2ecc71', label='Output'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8)

    plt.savefig(FIGS_DIR / 'architecture_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: architecture_pipeline.png")


def draw_search_detail():
    """Detailed search architecture."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5, 4.7, "FaceFlash Search Architecture", ha='center', fontsize=13, fontweight='bold')

    # Index (left side)
    rect = mpatches.FancyBboxPatch((0.3, 0.5), 3.5, 3.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#ffeaa7', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.05, 3.7, "INDEX", ha='center', fontsize=10, fontweight='bold')
    ax.text(2.05, 3.1, "Binary Codes (packed uint8)", ha='center', fontsize=8)
    ax.text(2.05, 2.6, "N × 64 bytes", ha='center', fontsize=8, color='#e74c3c')
    ax.text(2.05, 2.0, "Float Vectors (for rerank)", ha='center', fontsize=8)
    ax.text(2.05, 1.5, "N × 2048 bytes", ha='center', fontsize=8, color='#3498db')
    ax.text(2.05, 0.9, "PCA Projection Matrix", ha='center', fontsize=8)
    ax.text(2.05, 0.6, "512 × 512 float32", ha='center', fontsize=8, color='gray')

    # Search flow (right side)
    steps = [
        (6.0, 4.0, "Query Embedding (512-d float)"),
        (6.0, 3.2, "PCA Encode → 64-byte binary"),
        (6.0, 2.4, "Hamming Scan (Rust POPCNT)\nXOR + popcount all N vectors"),
        (6.0, 1.6, "Top-K Selection (argpartition)"),
        (6.0, 0.8, "Cosine Rerank (K dot products)"),
    ]

    for i, (x, y, text) in enumerate(steps):
        color = '#dfe6e9' if i == 0 else '#fab1a0' if i < 4 else '#55efc4'
        rect = mpatches.FancyBboxPatch((x-2.2, y-0.3), 4.4, 0.6,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)

    # Arrows between steps
    for i in range(len(steps) - 1):
        ax.annotate('', xy=(6.0, steps[i+1][1] + 0.3), xytext=(6.0, steps[i][1] - 0.3),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#2c3e50'))

    # Connection from index to search
    ax.annotate('', xy=(3.8, 2.4), xytext=(3.8, 2.4),
               arrowprops=dict(arrowstyle='->', lw=1))

    plt.savefig(FIGS_DIR / 'architecture_search.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: architecture_search.png")


def draw_memory_layout():
    """Memory layout comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # FAISS/HNSW layout
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("Traditional (FAISS/HNSW)", fontsize=11, fontweight='bold')
    ax.axis('off')

    rect = mpatches.FancyBboxPatch((0.5, 0.5), 9, 2.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#74b9ff', edgecolor='black')
    ax.add_patch(rect)
    ax.text(5, 2.5, "Float32 Vectors", ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 1.8, "N × 512 × 4 bytes = N × 2048 bytes", ha='center', fontsize=9)
    ax.text(5, 1.1, "100K faces = 195 MB", ha='center', fontsize=10, color='#d63031')

    # FaceFlash layout
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title("FaceFlash", fontsize=11, fontweight='bold')
    ax.axis('off')

    rect1 = mpatches.FancyBboxPatch((0.5, 1.8), 9, 1.2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#ff7675', edgecolor='black')
    ax.add_patch(rect1)
    ax.text(5, 2.5, "Binary Codes (search index)", ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 2.0, "N × 64 bytes → 100K = 6 MB", ha='center', fontsize=9)

    rect2 = mpatches.FancyBboxPatch((0.5, 0.3), 9, 1.2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#dfe6e9', edgecolor='black')
    ax.add_patch(rect2)
    ax.text(5, 1.0, "Float Vectors (rerank only)", ha='center', fontsize=9)
    ax.text(5, 0.5, "Only top-K accessed per query", ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'architecture_memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: architecture_memory.png")


if __name__ == "__main__":
    print("Generating architecture diagrams...")
    draw_pipeline()
    draw_search_detail()
    draw_memory_layout()
    print(f"\nAll figures saved to: {FIGS_DIR}")
