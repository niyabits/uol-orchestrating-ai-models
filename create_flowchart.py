import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with high DPI for professional quality
plt.figure(figsize=(12, 16))
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis('off')

# Define colors
primary_color = '#2E5EAA'  # Professional blue
secondary_color = '#F0F4F8'  # Light blue-gray
accent_color = '#E67E22'  # Orange for highlights
text_color = '#2C3E50'  # Dark blue-gray

# Helper function to create rounded rectangle boxes
def create_box(x, y, width, height, text, color=primary_color, text_color='white', fontsize=11):
    # Create rounded rectangle
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)

    # Add text
    ax.text(x, y, text, ha='center', va='center',
           fontsize=fontsize, color=text_color, weight='bold',
           wrap=True)

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color=text_color))

# Create boxes and arrows
y_positions = np.linspace(16.5, 1.5, 10)
x_center = 5

# Seed Prompt
create_box(x_center, y_positions[0], 3, 0.8, '[ Seed Prompt ]', accent_color)

# LLM Pipeline boxes
boxes = [
    ('LLM (Prompt Expansion)\nExpands seed into brief\n(Phi-3, Llama, Qwen)', secondary_color, text_color),
    ('LLM (Scene Structuring)\nOutline → JSON SceneSchema', secondary_color, text_color),
    ('LLM (Prose Rendering)\nScene plan → Narrative', secondary_color, text_color),
    ('LLM (Style Transfer)\nAdjust tone/style', secondary_color, text_color),
    ('LLM (Dialogue Generation)\nConflict-driven dialogue', secondary_color, text_color),
    ('Rule-based Sentiment Analyzer\nAffect / emotion tagging', primary_color, 'white'),
    ('Voice Synthesis Model\nSSML plan + NumPy preview', primary_color, 'white')
]

# Draw main pipeline boxes
for i, (text, color, txt_color) in enumerate(boxes):
    create_box(x_center, y_positions[i+1], 4.5, 1.2, text, color, txt_color, 10)
    if i < len(boxes):
        draw_arrow(x_center, y_positions[i] - 0.4, x_center, y_positions[i+1] + 0.6)

# Diffusion Models box (wider)
create_box(x_center, y_positions[8], 5, 1.4,
           'Diffusion Models\n• FLUX.1 Schnell (Scenes)\n• SDXL Realistic Vision\n  (Portraits)',
           primary_color, 'white', 10)
draw_arrow(x_center, y_positions[7] - 0.6, x_center, y_positions[8] + 0.7)

# Final output
create_box(x_center, y_positions[9], 4.5, 0.8, '[ Final Multimodal Story Bundle ]',
           accent_color, 'white', 12)
draw_arrow(x_center, y_positions[8] - 0.7, x_center, y_positions[9] + 0.4)

# Add title
plt.text(x_center, 17.5, 'Multimodal Story Generation Pipeline',
         ha='center', va='center', fontsize=16, weight='bold', color=text_color)

# Add subtle grid lines for professional look
for i in range(1, 10):
    plt.axhline(y=i*2, color='lightgray', linestyle='--', alpha=0.3, zorder=0)

# Tight layout and save
plt.tight_layout()
plt.savefig('multimodal_story_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('multimodal_story_pipeline.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Professional flowchart saved as 'multimodal_story_pipeline.png' and 'multimodal_story_pipeline.pdf'")
plt.show()