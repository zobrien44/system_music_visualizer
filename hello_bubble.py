
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk

TEXT = "Bonjour Mon Meilleur Amie"

BG = "#005BFF"          # blue background

FILL = "#FFFFFF"        # white text fill

OUTLINE = "#000000"     # black outline

OUTLINE_PX = 6          # ~6pt look; in raster rendering this is pixels

FONT_SIZE = 96

# Try common "bubble-ish" fonts on Windows; fall back if missing.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\coopbl.ttf",   # Cooper Black
    r"C:\Windows\Fonts\comic.ttf",    # Comic Sans MS
    r"C:\Windows\Fonts\arialbd.ttf",  # Arial Bold
]

def load_font():
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, FONT_SIZE)
        except OSError:
            pass
    return ImageFont.load_default()

font = load_font()

# Measure text
tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
dtmp = ImageDraw.Draw(tmp)
bbox = dtmp.textbbox((0, 0), TEXT, font=font, stroke_width=OUTLINE_PX)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

pad = 40
w, h = tw + pad * 2, th + pad * 2

# Render text with outline
img = Image.new("RGBA", (w, h), BG)
draw = ImageDraw.Draw(img)
x = (w - tw) // 2 - bbox[0]
y = (h - th) // 2 - bbox[1]
draw.text((x, y), TEXT, font=font, fill=FILL,
          stroke_width=OUTLINE_PX, stroke_fill=OUTLINE)

# Show in a separate window
root = tk.Tk()
root.title("Hello")
root.configure(bg=BG)

photo = ImageTk.PhotoImage(img)
label = tk.Label(root, image=photo, bg=BG, bd=0)
label.pack()

root.resizable(False, False)
root.mainloop()