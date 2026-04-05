import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model ----------------
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = LR().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

# ---------------- Image Transform ----------------
def smart_invert(x):
    return x if x.mean() < 0.5 else 1.0 - x

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(smart_invert),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------------- Prediction Logic ----------------
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Digit Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    preview = img.resize((180, 180))
    preview_tk = ImageTk.PhotoImage(preview)
    image_label.config(image=preview_tk)
    image_label.image = preview_tk

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    result_label.config(
        text=f"Prediction: {pred.item()}",
    )
    confidence_label.config(
        text=f"Confidence: {conf.item()*100:.2f}%"
    )

# ---------------- GUI Layout ----------------
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("420x520")
root.configure(bg="#f4f6f9")
root.resizable(False, False)

# Card Frame
card = tk.Frame(root, bg="white", bd=0, relief="flat")
card.place(relx=0.5, rely=0.5, anchor="center", width=360, height=460)

# Title
tk.Label(
    card,
    text="Handwritten Digit Recognition",
    font=("Segoe UI", 14, "bold"),
    bg="white"
).pack(pady=(20, 5))

tk.Label(
    card,
    text="Upload an image to predict the digit",
    font=("Segoe UI", 9),
    fg="#666",
    bg="white"
).pack(pady=(0, 15))

# Upload Button
tk.Button(
    card,
    text="Upload Image",
    command=upload_image,
    font=("Segoe UI", 10, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1e40af",
    relief="flat",
    padx=20,
    pady=8
).pack(pady=10)

# Image Preview
image_label = tk.Label(card, bg="white")
image_label.pack(pady=15)

# Result
result_label = tk.Label(
    card,
    text="Prediction: -",
    font=("Segoe UI", 12, "bold"),
    bg="white"
)
result_label.pack(pady=(10, 5))

confidence_label = tk.Label(
    card,
    text="Confidence: -",
    font=("Segoe UI", 10),
    fg="#444",
    bg="white"
)
confidence_label.pack(pady=(0, 20))

# Footer
tk.Label(
    card,
    text="PyTorch | MNIST | Tkinter GUI",
    font=("Segoe UI", 8),
    fg="#999",
    bg="white"
).pack(side="bottom", pady=10)

root.mainloop()
#jsdnsdnndsnjgit abaababab