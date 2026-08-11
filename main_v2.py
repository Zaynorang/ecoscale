import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from hx711v0_5_1 import HX711
import atexit
import math
from datetime import datetime
import threading

import firebase_admin
from firebase_admin import credentials, firestore

# ---------- Config ----------
MODEL_PATH = "my_model_ncnn_model"
CAM_INDEX = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

HARGA_JSON = "harga_sampah.json"
CONFIG_TIMBANGAN = "config_timbangan.json" 

WEIGHT_MAX_FILTER = 20000
CONF_THRESH = 0.5
FIREBASE_KEY = "firebase-key.json"  

# ---------- Load/Set Reference Unit ----------
REFERENCE_UNIT = 13.0
if os.path.exists(CONFIG_TIMBANGAN):
    try:
        with open(CONFIG_TIMBANGAN, "r") as f:
            cfg = json.load(f)
            REFERENCE_UNIT = cfg.get("reference_unit", 13.0)
            print(f"[INFO] Kalibrasi dimuat. Reference Unit: {REFERENCE_UNIT}")
    except Exception as e:
        print("[WARN] Gagal memuat file kalibrasi:", e)

# ---------- Firebase Setup ----------
if not firebase_admin._apps:
    if not os.path.exists(FIREBASE_KEY):
        raise SystemExit(f"[ERROR] File Firebase key tidak ditemukan: {FIREBASE_KEY}")
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred)
db = firestore.client()
print("[INFO] Firebase Firestore Terhubung (Data Teks)")

# ---------- Load harga JSON ----------
if not os.path.exists(HARGA_JSON):
    raise SystemExit(f"[ERROR] File harga tidak ditemukan: {HARGA_JSON}")

with open(HARGA_JSON, "r", encoding="utf-8") as f:
    harga_data = json.load(f)

label_to_info = {}
def perbarui_label_info():
    global label_to_info, BONCOS_HARGA_PER_KG
    label_to_info.clear()
    for kategori, data in harga_data.items():
        harga = data.get("harga_per_kg", 0)
        labels = data.get("labels", [])
        for lbl in labels:
            label_to_info[lbl] = (kategori, int(harga))
    BONCOS_HARGA_PER_KG = harga_data.get("boncos", {}).get("harga_per_kg", 0)

perbarui_label_info()
BONCOS_KATEGORI_KEY = "boncos"

daftar_kategori = list(harga_data.keys())
ITEMS_PER_PAGE = 12
total_pages = math.ceil(len(daftar_kategori) / ITEMS_PER_PAGE)
current_page = 0
cat_buttons_current = [] 

# ---------- Feedback & Input ----------
feedback_msg = ""
feedback_time = 0
FEEDBACK_DURATION = 2.0
input_numpad_str = ""
kategori_edit_terpilih = None

# ---------- HX711 ----------
hx = HX711(dout=5, pd_sck=26)
hx.setReferenceUnit(REFERENCE_UNIT)
try: hx.autosetOffset()
except Exception: pass

weight_history = []
last_valid_weight = 0.0

def read_weight_filtered():
    global last_valid_weight, weight_history
    try: val = hx.getWeight()
    except Exception: return last_valid_weight
    if val is None: return last_valid_weight
    try: val_f = float(val)
    except Exception: return last_valid_weight

    if val_f < 0: val_f = 0.0
    if val_f > WEIGHT_MAX_FILTER: return last_valid_weight

    weight_history.append(val_f)
    if len(weight_history) > 15: 
        weight_history.pop(0)
    
    if len(weight_history) >= 5:
        sorted_history = sorted(weight_history)
        trim_idx = len(sorted_history) // 5
        valid_samples = sorted_history[trim_idx : -trim_idx]
        smoothed = sum(valid_samples) / len(valid_samples)
    else:
        smoothed = sum(weight_history) / len(weight_history)
    
    # Pembulatan kelipatan 5 gram agar layar tenang
    smoothed = round(smoothed / 5.0) * 5.0
    last_valid_weight = smoothed
    return smoothed

def hitung_harga(berat_kg, harga_per_kg):
    berat_bulat = math.floor(berat_kg * 2) / 2
    if berat_bulat < 0.5: berat_bulat = 0.5
    return berat_bulat * harga_per_kg

# ---------- YOLO ----------
model = YOLO(MODEL_PATH, task="detect")
labels_map = model.names

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]

def cleanup():
    try: cap.release()
    except: pass
    cv2.destroyAllWindows()
atexit.register(cleanup)

# ---------- State Application & UI Buttons ----------
app_state = "MENU"
click_action = None

# Menu Utama (5 Baris)
BTN_MENU_SINGLE = (120, 70, 400, 50)
BTN_MENU_LIVE   = (120, 140, 400, 50)
BTN_MENU_CALIB  = (120, 210, 400, 50)
BTN_MENU_EDIT_P = (120, 280, 400, 50)
BTN_MENU_EXIT   = (120, 350, 400, 50)

# Tombol Mode Kamera
BTN_SAVE  = (220, VIDEO_HEIGHT - 60, 150, 40)
BTN_SCAN  = (220, VIDEO_HEIGHT - 60, 150, 40)
BTN_RETRY = (380, VIDEO_HEIGHT - 60, 120, 40)
BTN_EXIT  = (10, VIDEO_HEIGHT - 60, 100, 40)
BTN_BACK  = (10, 10, 100, 40) 
BTN_TARE  = (520, 10, 110, 40)
BTN_EDIT  = (10, 230, 200, 40) 

# Tombol Navigasi Bawah
BTN_PREV       = (20, 420, 120, 40)
BTN_CANCEL_CAT = (260, 420, 120, 40)
BTN_NEXT       = (500, 420, 120, 40)

# Tombol Universal
BTN_CALIB_TARE   = (220, 220, 200, 60)
BTN_CALIB_CANCEL = (220, 320, 200, 50)
BTN_GEN_BACK     = (10, 10, 100, 40)

# Keyboard Numpad Virtual (3x4)
numpad_buttons = [
    ("1", 160, 180), ("2", 260, 180), ("3", 360, 180),
    ("4", 160, 240), ("5", 260, 240), ("6", 360, 240),
    ("7", 160, 300), ("8", 260, 300), ("9", 360, 300),
    ("CLR", 160, 360), ("0", 260, 360), ("OK", 360, 360)
]
NUMPAD_W, NUMPAD_H = 80, 50

def is_inside(x, y, btn):
    bx, by, bw, bh = btn
    return bx <= x <= bx + bw and by <= y <= by + bh

def mouse_callback(event, x, y, flags, param):
    global click_action, kategori_terpilih_manual, app_state
    if event == cv2.EVENT_LBUTTONDOWN:
        if app_state == "MENU":
            if is_inside(x, y, BTN_MENU_SINGLE): click_action = "GOTO_SINGLE"
            elif is_inside(x, y, BTN_MENU_LIVE): click_action = "GOTO_LIVE"
            elif is_inside(x, y, BTN_MENU_CALIB): click_action = "GOTO_CALIB"
            elif is_inside(x, y, BTN_MENU_EDIT_P): click_action = "GOTO_EDIT_PRICE"
            elif is_inside(x, y, BTN_MENU_EXIT): click_action = "DO_EXIT"
        
        elif app_state in ["LIVE", "SINGLE_CAM"]:
            if is_inside(x, y, BTN_EXIT): click_action = "DO_EXIT"
            elif is_inside(x, y, BTN_BACK): click_action = "GOTO_MENU"
            elif is_inside(x, y, BTN_TARE): click_action = "DO_TARE"
            elif app_state == "LIVE" and is_inside(x, y, BTN_SAVE): click_action = "DO_SAVE"
            elif app_state == "SINGLE_CAM" and is_inside(x, y, BTN_SCAN): click_action = "DO_SCAN"
        
        elif app_state == "SINGLE_RESULT":
            if is_inside(x, y, BTN_SAVE): click_action = "DO_SAVE"
            elif is_inside(x, y, BTN_RETRY): click_action = "DO_RETRY"
            elif is_inside(x, y, BTN_BACK): click_action = "GOTO_MENU"
            elif is_inside(x, y, BTN_EDIT): click_action = "DO_EDIT"
            
        elif app_state in ["SELECT_CATEGORY", "EDIT_PRICE_MENU"]:
            for btn in cat_buttons_current:
                if is_inside(x, y, btn["rect"]):
                    kategori_terpilih_manual = btn["kat"]
                    click_action = "KATEGORI_TERPILIH"
                    break
            
            if is_inside(x, y, BTN_PREV) and current_page > 0: click_action = "PREV_PAGE"
            elif is_inside(x, y, BTN_NEXT) and current_page < total_pages - 1: click_action = "NEXT_PAGE"
            elif is_inside(x, y, BTN_CANCEL_CAT): click_action = "CANCEL_ACT"

        elif app_state == "CALIB_STEP1":
            if is_inside(x, y, BTN_CALIB_TARE): click_action = "DO_CALIB_TARE"
            elif is_inside(x, y, BTN_CALIB_CANCEL): click_action = "GOTO_MENU"

        elif app_state in ["CALIB_STEP2", "EDIT_PRICE_INPUT"]:
            if is_inside(x, y, BTN_GEN_BACK): click_action = "GO_BACK"
            else:
                for label, bx, by in numpad_buttons:
                    if bx <= x <= bx + NUMPAD_W and by <= y <= by + NUMPAD_H:
                        click_action = f"NUMPAD_{label}"
                        break

cv2.namedWindow("EcoScale", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("EcoScale", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("EcoScale", mouse_callback)

def save_data_to_firestore(payload):
    try:
        db.collection("riwayat_prediksi").add(payload)
        print(f"[FIREBASE] Data tersimpan!")
    except Exception as e:
        print("[ERROR] Gagal simpan ke Firestore:", e)

single_scan_payload = None
single_freeze_frame = None

# ---------- Loop Utama ----------
while True:
    ret, frame = cap.read()
    if not ret: continue

    weight_g = read_weight_filtered()
    weight_kg = weight_g / 1000.0

    # ----- 1. MENU UTAMA -----
    if app_state == "MENU":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (VIDEO_WIDTH, VIDEO_HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame, "ECOSCALE VISION", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        sx, sy, sw, sh = BTN_MENU_SINGLE
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 100, 0), -1)
        cv2.putText(frame, "Scan 1 Kali (Manual)", (sx+30, sy+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        lx, ly, lw, lh = BTN_MENU_LIVE
        cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0, 180, 255), -1)
        cv2.putText(frame, "Live Preview (Otomatis)", (lx+30, ly+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        cx, cy, cw, ch = BTN_MENU_CALIB
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (255, 0, 255), -1)
        cv2.putText(frame, "Kalibrasi Timbangan", (cx+30, cy+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        epx, epy, epw, eph = BTN_MENU_EDIT_P
        cv2.rectangle(frame, (epx, epy), (epx+epw, epy+eph), (150, 150, 0), -1)
        cv2.putText(frame, "Edit Harga Kategori", (epx+30, epy+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        ex, ey, ew, eh = BTN_MENU_EXIT
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), -1)
        cv2.putText(frame, "Keluar Aplikasi", (ex+100, ey+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if click_action == "GOTO_SINGLE": app_state = "SINGLE_CAM"; click_action = None
        elif click_action == "GOTO_LIVE": app_state = "LIVE"; click_action = None
        elif click_action == "GOTO_CALIB": app_state = "CALIB_STEP1"; click_action = None
        elif click_action == "GOTO_EDIT_PRICE": 
            app_state = "EDIT_PRICE_MENU"
            current_page = 0
            click_action = None
        elif click_action == "DO_EXIT": break

    # ----- 2. KALIBRASI -----
    elif app_state == "CALIB_STEP1":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (VIDEO_WIDTH, VIDEO_HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        cv2.putText(frame, "KALIBRASI: LANGKAH 1 DARI 2", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, "1. Singkirkan SEMUA barang dari atas timbangan.", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "2. Tekan tombol TARE di bawah untuk menetapkan", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "   titik Nol (0).", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        tx, ty, tw, th = BTN_CALIB_TARE
        cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (0, 200, 255), -1)
        cv2.putText(frame, "TARE (NOL-KAN)", (tx+10, ty+38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cx, cy, cw, ch = BTN_CALIB_CANCEL
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), -1)
        cv2.putText(frame, "BATAL", (cx+60, cy+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if click_action == "DO_CALIB_TARE":
            try:
                hx.autosetOffset()
                weight_history.clear()
                last_valid_weight = 0.0
                input_numpad_str = ""
                app_state = "CALIB_STEP2"
            except Exception as e: print("Gagal Tare:", e)
            click_action = None
        elif click_action == "GOTO_MENU":
            app_state = "MENU"; click_action = None

    elif app_state == "CALIB_STEP2":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (VIDEO_WIDTH, VIDEO_HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        bx, by, bw, bh = BTN_GEN_BACK
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
        cv2.putText(frame, "BATAL", (bx+20, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, "KALIBRASI: LANGKAH 2 DARI 2", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "Taruh benda referensi. Ketik beratnya (GRAM):", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.rectangle(frame, (160, 135), (440, 175), (255,255,255), -1)
        display_text = input_numpad_str + " g" if input_numpad_str else "0 g"
        cv2.putText(frame, display_text, (170, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        for label, nx, ny in numpad_buttons:
            color, text_col = (100, 100, 100), (255,255,255)
            if label == "CLR": color = (0, 0, 255)
            elif label == "OK": color = (0, 200, 0); text_col = (0,0,0)
            cv2.rectangle(frame, (nx, ny), (nx+NUMPAD_W, ny+NUMPAD_H), color, -1)
            tx = nx + 25 if len(label) == 1 else nx + 15
            cv2.putText(frame, label, (tx, ny+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_col, 2)

        if click_action and click_action.startswith("NUMPAD_"):
            val = click_action.split("_")[1]
            if val == "CLR": input_numpad_str = ""
            elif val == "OK":
                if input_numpad_str:
                    target_g = float(input_numpad_str)
                    if target_g > 0:
                        try:
                            hx.setReferenceUnit(1)
                            raw_val = hx.getWeight()
                            new_ref = raw_val / target_g
                            hx.setReferenceUnit(new_ref)
                            with open(CONFIG_TIMBANGAN, "w") as f:
                                json.dump({"reference_unit": new_ref}, f)
                            feedback_msg = "Kalibrasi Berhasil!"
                            feedback_time = time.time()
                        except Exception as e: print("Gagal kalibrasi:", e)
                    app_state = "MENU"
            else: input_numpad_str += val
            click_action = None

        elif click_action == "GO_BACK": app_state = "MENU"; click_action = None

    # ----- 3. MENU EDIT HARGA -----
    elif app_state == "EDIT_PRICE_MENU":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (VIDEO_WIDTH, VIDEO_HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        cv2.putText(frame, f"PILIH KATEGORI UTK DIEDIT (Hal {current_page+1}/{total_pages}):", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        start_idx = current_page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(daftar_kategori))
        cat_buttons_current.clear()
        
        for i in range(start_idx, end_idx):
            idx = i - start_idx
            bx, by = 20 + ((idx % 2) * 310), 90 + ((idx // 2) * 50)
            bw, bh = 290, 40
            cat_buttons_current.append({"kat": daftar_kategori[i], "rect": (bx, by, bw, bh)})
            
            # Highlight border to show it's edit mode
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (150, 150, 0), -1)
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 0), 2)
            
            hrg = harga_data.get(daftar_kategori[i], {}).get("harga_per_kg", 0)
            teks = f"{daftar_kategori[i].upper()[:12]}: Rp{hrg}"
            cv2.putText(frame, teks, (bx+10, by+26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if current_page > 0:
            px, py, pw, ph = BTN_PREV
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 150, 0), -1)
            cv2.putText(frame, "< PREV", (px+15, py+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if current_page < total_pages - 1:
            nx, ny, nw, nh = BTN_NEXT
            cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (255, 150, 0), -1)
            cv2.putText(frame, "NEXT >", (nx+15, ny+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cx, cy, cw, ch = BTN_CANCEL_CAT
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), -1)
        cv2.putText(frame, "KEMBALI", (cx+15, cy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if click_action == "KATEGORI_TERPILIH":
            kategori_edit_terpilih = kategori_terpilih_manual
            input_numpad_str = ""
            app_state = "EDIT_PRICE_INPUT"
            click_action = None
        elif click_action == "PREV_PAGE": current_page -= 1; click_action = None
        elif click_action == "NEXT_PAGE": current_page += 1; click_action = None
        elif click_action == "CANCEL_ACT": app_state = "MENU"; click_action = None

    elif app_state == "EDIT_PRICE_INPUT":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (VIDEO_WIDTH, VIDEO_HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        bx, by, bw, bh = BTN_GEN_BACK
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
        cv2.putText(frame, "BATAL", (bx+20, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        hrg_lama = harga_data.get(kategori_edit_terpilih, {}).get("harga_per_kg", 0)
        cv2.putText(frame, f"UBAH HARGA: {kategori_edit_terpilih.upper()}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Harga lama: Rp{hrg_lama}/kg. Ketik harga baru:", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.rectangle(frame, (160, 135), (440, 175), (255,255,255), -1)
        display_text = "Rp " + (input_numpad_str if input_numpad_str else "0")
        cv2.putText(frame, display_text, (170, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        for label, nx, ny in numpad_buttons:
            color, text_col = (100, 100, 100), (255,255,255)
            if label == "CLR": color = (0, 0, 255)
            elif label == "OK": color = (0, 200, 0); text_col = (0,0,0)
            cv2.rectangle(frame, (nx, ny), (nx+NUMPAD_W, ny+NUMPAD_H), color, -1)
            tx = nx + 25 if len(label) == 1 else nx + 15
            cv2.putText(frame, label, (tx, ny+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_col, 2)

        if click_action and click_action.startswith("NUMPAD_"):
            val = click_action.split("_")[1]
            if val == "CLR": input_numpad_str = ""
            elif val == "OK":
                if input_numpad_str:
                    harga_baru = int(input_numpad_str)
                    # Simpan ke memori dictionary
                    if kategori_edit_terpilih not in harga_data:
                        harga_data[kategori_edit_terpilih] = {"harga_per_kg": 0, "labels": []}
                    harga_data[kategori_edit_terpilih]["harga_per_kg"] = harga_baru
                    
                    # Simpan permanen ke file json dengan format rapi (indent)
                    try:
                        with open(HARGA_JSON, "w", encoding="utf-8") as f:
                            json.dump(harga_data, f, indent=4)
                        perbarui_label_info()
                        feedback_msg = f"Harga {kategori_edit_terpilih} Disimpan!"
                        feedback_time = time.time()
                    except Exception as e: print("Gagal save JSON:", e)
                
                app_state = "EDIT_PRICE_MENU"
            else: input_numpad_str += val
            click_action = None

        elif click_action == "GO_BACK": app_state = "EDIT_PRICE_MENU"; click_action = None


    # ----- 4. LIVE ATAU STANDBY SINGLE CAM -----
    elif app_state in ["LIVE", "SINGLE_CAM"]:
        if click_action == "DO_TARE":
            try:
                hx.autosetOffset()
                weight_history.clear()
                last_valid_weight = 0.0
                feedback_msg = "Timbangan di-TARE!"
                feedback_time = time.time()
            except Exception: pass
            click_action = None
            
        detected_label, detected_kategori = "-", "-"
        harga_per_kg, harga_total, best_conf = 0, 0, 0.0
        is_boncos = False
        
        if app_state == "LIVE" or (app_state == "SINGLE_CAM" and click_action == "DO_SCAN"):
            results = model(frame, verbose=False)
            unique_kat = set()
            best_det = None
            
            for det in results[0].boxes:
                conf = float(det.conf.item())
                if conf < CONF_THRESH: continue
                cls_idx = int(det.cls.item())
                label_name = labels_map.get(cls_idx, str(cls_idx))
                
                if app_state == "LIVE":
                    xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                    color = bbox_colors[cls_idx % len(bbox_colors)]
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, f"{label_name}", (xyxy[0], xyxy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                info_tmp = label_to_info.get(label_name)
                if info_tmp: unique_kat.add(info_tmp[0])
                if conf > best_conf:
                    best_conf = conf
                    best_det = {"label": label_name}

            if len(unique_kat) > 1:
                is_boncos = True
                detected_label, detected_kategori, harga_per_kg = "boncos", BONCOS_KATEGORI_KEY, BONCOS_HARGA_PER_KG
            elif best_det:
                detected_label = best_det["label"]
                info = label_to_info.get(detected_label)
                if info: detected_kategori, harga_per_kg = info
            
            harga_total = hitung_harga(weight_kg, harga_per_kg) if weight_kg > 0 else 0

        if app_state == "SINGLE_CAM" and click_action == "DO_SCAN":
            single_scan_payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": detected_label, "kategori": detected_kategori,
                "berat_gram": round(weight_g, 1), "berat_kg": round(weight_kg, 3),
                "harga_per_kg": harga_per_kg, "total": harga_total,
                "confidence": round(best_conf, 3), "is_boncos": is_boncos
            }
            app_state = "SINGLE_RESULT"
            click_action = None
            single_freeze_frame = frame.copy()
            continue

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (360, 220), (0,0,0), -1) 
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        y0, dy = 76, 26
        cv2.putText(frame, f"Berat: {weight_g:.1f} g", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Label: {detected_label}", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Kategori: {detected_kategori}", (20, y0+dy*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Harga/kg: Rp{harga_per_kg}", (20, y0+dy*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.putText(frame, f"Total: Rp{harga_total}", (20, y0+dy*4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        bx, by, bw, bh = BTN_BACK
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
        cv2.putText(frame, "MENU", (bx+15, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        ex, ey, ew, eh = BTN_EXIT
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,0,255), -1)
        cv2.putText(frame, "EXIT", (ex+15, ey+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        tx, ty, tw, th = BTN_TARE
        cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (0, 200, 255), -1)
        cv2.putText(frame, "TARE", (tx+25, ty+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        if app_state == "LIVE":
            sx, sy, sw, sh = BTN_SAVE
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 180, 255), -1)
            cv2.putText(frame, "SAVE", (sx+35, sy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            if click_action == "DO_SAVE":
                payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": detected_label, "kategori": detected_kategori,
                    "berat_gram": round(weight_g, 1), "berat_kg": round(weight_kg, 3),
                    "harga_per_kg": harga_per_kg, "total": harga_total,
                    "confidence": round(best_conf, 3), "is_boncos": is_boncos
                }
                threading.Thread(target=save_data_to_firestore, args=(payload,), daemon=True).start()
                feedback_msg = "Data Tersimpan!"
                feedback_time = time.time()
                click_action = None
        
        elif app_state == "SINGLE_CAM":
            scx, scy, scw, sch = BTN_SCAN
            cv2.rectangle(frame, (scx, scy), (scx+scw, scy+sch), (255, 0, 0), -1)
            cv2.putText(frame, "SCAN", (scx+35, scy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if click_action == "GOTO_MENU": app_state = "MENU"; click_action = None
        if click_action == "DO_EXIT": break

    # ----- 5. HASIL SCAN SINGLE CAM (BEKU) -----
    elif app_state == "SINGLE_RESULT":
        frame = single_freeze_frame.copy()
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (360, 220), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        lbl = single_scan_payload["label"]
        kat = single_scan_payload["kategori"]
        w_g = single_scan_payload["berat_gram"]
        h_kg = single_scan_payload["harga_per_kg"]
        tot = single_scan_payload["total"]

        y0, dy = 76, 26
        cv2.putText(frame, f"Berat: {w_g:.1f} g", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Label: {lbl}", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Kategori: {kat}", (20, y0+dy*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Harga/kg: Rp{h_kg}", (20, y0+dy*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.putText(frame, f"Total: Rp{tot}", (20, y0+dy*4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        bx, by, bw, bh = BTN_BACK
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
        cv2.putText(frame, "MENU", (bx+15, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        sx, sy, sw, sh = BTN_SAVE
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 180, 255), -1)
        cv2.putText(frame, "SAVE", (sx+35, sy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        rx, ry, rw, rh = BTN_RETRY
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), -1)
        cv2.putText(frame, "RETRY", (rx+15, ry+28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        edx, edy, edw, edh = BTN_EDIT
        cv2.rectangle(frame, (edx, edy), (edx+edw, edy+edh), (255, 0, 255), -1)
        cv2.putText(frame, "GANTI KATEGORI", (edx+10, edy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if click_action == "DO_EDIT": app_state = "SELECT_CATEGORY"; click_action = None
        elif click_action == "DO_SAVE":
            threading.Thread(target=save_data_to_firestore, args=(single_scan_payload,), daemon=True).start()
            feedback_msg = "Data Tersimpan!"
            feedback_time = time.time()
            app_state = "SINGLE_CAM"; click_action = None
        elif click_action == "DO_RETRY": app_state = "SINGLE_CAM"; click_action = None
        elif click_action == "GOTO_MENU": app_state = "MENU"; click_action = None

    # ----- 6. PILIH KATEGORI MANUAL (PAGINATION) -----
    elif app_state == "SELECT_CATEGORY":
        frame = single_freeze_frame.copy()
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (VIDEO_WIDTH, VIDEO_HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        cv2.putText(frame, f"PILIH KATEGORI (Hal {current_page+1}/{total_pages}):", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        start_idx = current_page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(daftar_kategori))
        cat_buttons_current.clear()
        
        for i in range(start_idx, end_idx):
            idx_in_page = i - start_idx
            bx, by = 20 + ((idx_in_page % 2) * 310), 90 + ((idx_in_page // 2) * 50)
            bw, bh = 290, 40
            cat_buttons_current.append({"kat": daftar_kategori[i], "rect": (bx, by, bw, bh)})
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (80, 80, 80), -1)
            display_text = daftar_kategori[i].upper().replace("_", " ")
            cv2.putText(frame, display_text, (bx+10, by+26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if current_page > 0:
            px, py, pw, ph = BTN_PREV
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 150, 0), -1)
            cv2.putText(frame, "< PREV", (px+15, py+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if current_page < total_pages - 1:
            nx, ny, nw, nh = BTN_NEXT
            cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (255, 150, 0), -1)
            cv2.putText(frame, "NEXT >", (nx+15, ny+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cx, cy, cw, ch = BTN_CANCEL_CAT
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), -1)
        cv2.putText(frame, "CANCEL", (cx+15, cy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if click_action == "KATEGORI_TERPILIH":
            single_scan_payload["kategori"] = kategori_terpilih_manual
            single_scan_payload["label"] = "(Diubah Manual)"
            if kategori_terpilih_manual == "boncos": single_scan_payload["harga_per_kg"] = BONCOS_HARGA_PER_KG
            else: single_scan_payload["harga_per_kg"] = harga_data.get(kategori_terpilih_manual, {}).get("harga_per_kg", 0)
            
            w_kg = single_scan_payload["berat_kg"]
            single_scan_payload["total"] = hitung_harga(w_kg, single_scan_payload["harga_per_kg"])
            app_state = "SINGLE_RESULT"; click_action = None
        
        elif click_action == "PREV_PAGE": current_page -= 1; click_action = None
        elif click_action == "NEXT_PAGE": current_page += 1; click_action = None
        elif click_action == "CANCEL_ACT": app_state = "SINGLE_RESULT"; click_action = None

    # ----- 7. RENDER FEEDBACK OVERLAY GLOBAL -----
    if feedback_msg and time.time() - feedback_time < FEEDBACK_DURATION:
        # Tampilkan kotak hijau semi transparan di tengah layar
        overlay = frame.copy()
        box_w, box_h = 400, 60
        tx, ty = (VIDEO_WIDTH - box_w) // 2, (VIDEO_HEIGHT - box_h) // 2
        cv2.rectangle(overlay, (tx, ty), (tx + box_w, ty + box_h), (0, 200, 0), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        # Teks diletakkan agak di tengah kotak
        cv2.putText(frame, feedback_msg, (tx + 40, ty + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("EcoScale", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
