from panorama import Stitcher
import imutils
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

# --- Hàm resize ảnh để hiển thị trong GUI ---
def cv2_to_tk(img, maxsize=(600, 400)):
    h, w = img.shape[:2]
    # Tính tỉ lệ scale để ảnh vừa khung
    scale = min(maxsize[0] / w, maxsize[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))

# --- Hàm chọn ảnh ---
def select_image1():
    global imageA
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return
    imageA = cv2.imread(path)
    if imageA is None:
        messagebox.showerror("Lỗi", "Không thể đọc ảnh 1.")
        return
    tk_img = cv2_to_tk(imageA)
    lbl_img1.config(image=tk_img)
    lbl_img1.image = tk_img

def select_image2():
    global imageB
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return
    imageB = cv2.imread(path)
    if imageB is None:
        messagebox.showerror("Lỗi", "Không thể đọc ảnh 2.")
        return
    tk_img = cv2_to_tk(imageB)
    lbl_img2.config(image=tk_img)
    lbl_img2.image = tk_img

# --- Hàm ghép ảnh ---
def stitch_images():
    global result, vis, imageA_key, imageB_key
    if imageA is None or imageB is None:
        messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn đủ 2 ảnh trước khi ghép.")
        return
    stitcher = Stitcher()
    output = stitcher.stitch([imutils.resize(imageA, width=400),imutils.resize(imageB, width=400)],showMatches=True)
    if output is None:
        messagebox.showerror("Thất bại", "Không thể ghép 2 ảnh.")
        return
    (result, vis, imageA_key, imageB_key) = output
    result = crop_black(result)

    # Hiển thị kết quả
    img1_tk = cv2_to_tk(imageA_key)
    img2_tk = cv2_to_tk(imageB_key)
    vis_tk = cv2_to_tk(vis, maxsize=(800, 300))
    res_tk = cv2_to_tk(result, maxsize=(800, 300))
    lbl_keyA.config(image=img1_tk)
    lbl_keyA.image = img1_tk
    lbl_keyB.config(image=img2_tk)
    lbl_keyB.image = img2_tk
    lbl_vis.config(image=vis_tk)
    lbl_vis.image = vis_tk
    lbl_result.config(image=res_tk)
    lbl_result.image = res_tk
    messagebox.showinfo("Thành công", "Ghép ảnh hoàn tất!")

# --- Cắt bỏ vùng đen (pixel = 0) quanh ảnh sau khi ghép Panorama, giữ lại vùng có nội dung thật ---
def crop_black(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tạo mask: vùng có giá trị > 0 là có nội dung
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped

# --- Hàm lưu ảnh panorama ---
def save_panorama():
    if result is None:
        messagebox.showwarning("Chưa có ảnh", "Hãy ghép ảnh trước khi lưu.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg",filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],title="Lưu ảnh panorama")
    if save_path:
        cv2.imwrite(save_path, result)
        messagebox.showinfo("Đã lưu", f"Ảnh panorama đã được lưu tại:\n{save_path}")

# --- Giao diện chính ---
root = tk.Tk()
root.title("Ghép Ảnh Panorama")
root.geometry("1260x800")
root.configure(bg="#F3F3F3")
imageA = imageB = result = vis = imageA_key = imageB_key = None
imageA_path = imageB_path = ""

# --- Hàng chọn ảnh ---
frm_top = tk.Frame(root, bg="#F3F3F3")
frm_top.pack(pady=10)
btn_select1 = tk.Button(frm_top, text="📂 Chọn ảnh 1", command=select_image1, width=15, bg="#4CAF50", fg="white")
btn_select1.grid(row=0, column=0, padx=10)
btn_select2 = tk.Button(frm_top, text="📂 Chọn ảnh 2", command=select_image2, width=15, bg="#4CAF50", fg="white")
btn_select2.grid(row=0, column=1, padx=10)
btn_stitch = tk.Button(frm_top, text="🧩 Ghép ảnh", command=stitch_images, width=15, bg="#2196F3", fg="white")
btn_stitch.grid(row=0, column=2, padx=10)
btn_save = tk.Button(frm_top, text="💾 Lưu ảnh Panorama", command=save_panorama, width=18, bg="#9C27B0", fg="white")
btn_save.grid(row=0, column=3, padx=10)

# --- Canvas + Scrollbar để cuộn nội dung ---
canvas = tk.Canvas(root, bg="#F3F3F3", highlightthickness=0)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# Tạo frame chứa nội dung bên trong Canvas
frm_images = tk.Frame(canvas, bg="#F3F3F3")
canvas.create_window((0, 0), window=frm_images, anchor="nw")

# Tự động cập nhật vùng cuộn khi kích thước thay đổi
def update_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
frm_images.bind("<Configure>", update_scrollregion)

# --- Các frame hiển thị ảnh ---
# Ảnh gốc
frm_img1 = tk.Frame(frm_images, width=600, height=400, bg="#CCCCCC")
frm_img1.grid(row=0, column=0, padx=10, pady=10)
frm_img1.pack_propagate(False)
lbl_img1 = tk.Label(frm_img1, text="Ảnh 1", bg="#CCCCCC")
lbl_img1.pack(expand=True)

frm_img2 = tk.Frame(frm_images, width=600, height=400, bg="#CCCCCC")
frm_img2.grid(row=0, column=1, padx=10, pady=10)
frm_img2.pack_propagate(False)
lbl_img2 = tk.Label(frm_img2, text="Ảnh 2", bg="#CCCCCC")
lbl_img2.pack(expand=True)

# Keypoints
frm_keyA = tk.Frame(frm_images, width=600, height=400, bg="#CCCCCC")
frm_keyA.grid(row=1, column=0, padx=10, pady=10)
frm_keyA.pack_propagate(False)
lbl_keyA = tk.Label(frm_keyA, text="Keypoints Ảnh 1", bg="#CCCCCC")
lbl_keyA.pack(expand=True)

frm_keyB = tk.Frame(frm_images, width=600, height=400, bg="#CCCCCC")
frm_keyB.grid(row=1, column=1, padx=10, pady=10)
frm_keyB.pack_propagate(False)
lbl_keyB = tk.Label(frm_keyB, text="Keypoints Ảnh 2", bg="#CCCCCC")
lbl_keyB.pack(expand=True)

# Matches
frm_vis = tk.Frame(frm_images, width=1200, height=400, bg="#CCCCCC")
frm_vis.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
frm_vis.pack_propagate(False)
lbl_vis = tk.Label(frm_vis, text="Matches", bg="#CCCCCC")
lbl_vis.pack(expand=True)

# Panorama
frm_result = tk.Frame(frm_images, width=1200, height=400, bg="#CCCCCC")
frm_result.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
frm_result.pack_propagate(False)
lbl_result = tk.Label(frm_result, text="Ảnh Panorama", bg="#CCCCCC")
lbl_result.pack(expand=True)

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

root.mainloop()