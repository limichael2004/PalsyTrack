import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
from datetime import datetime

#Configuration

class Config:
    APP_NAME = "PalsyTrack by Michael Li"
    SUBTITLE = "Clinical Research Dashboard v7.2"
    DATA_DIR = "palsytrack_data"
    IMAGE_DIR = os.path.join(DATA_DIR, "captures")
    CSV_PATH = os.path.join(DATA_DIR, "biometrics.csv")
    
    # Visuals
    THEME_BG = "#101010"
    THEME_PANEL = "#1e1e1e" 
    THEME_TEXT = "#e0e0e0"
    THEME_ACCENT = "#00ffcc" # Medical Cyan
    THEME_ALERT = "#ff3333"
    THEME_WARN = "#ffaa00"
    THEME_ACTIVE = "#ffff00" # Yellow for selected ghost feature
    
    # Camera
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    
    # --- ANATOMICAL INDICES (Face Mesh) ---
    L_EYE_IDX = [33, 159, 133, 145] 
    R_EYE_IDX = [362, 386, 263, 374]
    L_PUPIL = 468
    R_PUPIL = 473
    L_BROW = [70, 105, 46]
    R_BROW = [300, 334, 276]
    
    L_EYE_CORNERS = [133, 33]  # Inner, Outer
    R_EYE_CORNERS = [362, 263] # Inner, Outer
    
    # "Ears" / Face Width Markers (Tragion area)
    L_EAR_IDX = [234, 93, 132] 
    R_EAR_IDX = [454, 323, 361]
    
    L_MOUTH_CORNER = 61
    R_MOUTH_CORNER = 291
    
    # QC
    MAX_YAW = 6.0
    MAX_ROLL = 4.0

os.makedirs(Config.IMAGE_DIR, exist_ok=True)

#biomath

class BioMath:
    @staticmethod
    def polygon_area(points_3d):
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def calculate_tilt(p_inner, p_outer):
        dy = p_outer[1] - p_inner[1]
        dx = p_outer[0] - p_inner[0]
        return np.degrees(np.arctan2(-dy, dx))

class FacialEngine:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None, {"status": "NO_FACE"}
            
        lms = results.multi_face_landmarks[0]
        mesh = np.array([np.array([p.x*w, p.y*h, p.z*w]) for p in lms.landmark])
        
        # QC: Head Pose
        p1 = mesh[33]
        p2 = mesh[263]
        dy, dx = p2[1]-p1[1], p2[0]-p1[0]
        roll = np.degrees(np.arctan2(dy, dx))
        
        nose = mesh[1]
        mid_eye = (p1 + p2) / 2
        yaw = (nose[0] - mid_eye[0]) * 0.5 
        
        qc = {
            "status": "OK",
            "yaw": yaw, 
            "roll": roll,
            "lighting": np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        }
        return mesh, qc

    def extract_metrics(self, mesh):
        l_pupil = mesh[Config.L_PUPIL]
        r_pupil = mesh[Config.R_PUPIL]
        ipd = np.linalg.norm(l_pupil - r_pupil)
        if ipd == 0: ipd = 1.0
        
        def n_dist(i1, i2): return np.linalg.norm(mesh[i1]-mesh[i2]) / ipd

        m = {}
        m['L_Eye_H'] = n_dist(159, 145)
        m['R_Eye_H'] = n_dist(386, 374)
        m['L_Eye_W'] = n_dist(33, 133)
        m['R_Eye_W'] = n_dist(362, 263)
        
        m['L_Canthal_Tilt'] = BioMath.calculate_tilt(mesh[Config.L_EYE_CORNERS[0]], mesh[Config.L_EYE_CORNERS[1]])
        m['R_Canthal_Tilt'] = BioMath.calculate_tilt(mesh[Config.R_EYE_CORNERS[0]], mesh[Config.R_EYE_CORNERS[1]])
        
        m['L_Brow_Elev'] = n_dist(Config.L_PUPIL, 105) 
        m['R_Brow_Elev'] = n_dist(Config.R_PUPIL, 334)
        
        l_ear_pts = mesh[Config.L_EAR_IDX]
        r_ear_pts = mesh[Config.R_EAR_IDX]
        m['L_Ear_Area'] = BioMath.polygon_area(l_ear_pts) / (ipd**2)
        m['R_Ear_Area'] = BioMath.polygon_area(r_ear_pts) / (ipd**2)
        
        m['L_Cnr_Drop'] = (mesh[Config.L_MOUTH_CORNER][1] - mesh[Config.L_PUPIL][1]) / ipd
        m['R_Cnr_Drop'] = (mesh[Config.R_MOUTH_CORNER][1] - mesh[Config.R_PUPIL][1]) / ipd
        
        dy = mesh[Config.R_MOUTH_CORNER][1] - mesh[Config.L_MOUTH_CORNER][1]
        dx = mesh[Config.R_MOUTH_CORNER][0] - mesh[Config.L_MOUTH_CORNER][0]
        m['Mouth_Slant'] = np.degrees(np.arctan2(dy, dx))
        
        m['Asym_Eye_Open'] = abs(m['L_Eye_H'] - m['R_Eye_H'])
        m['Asym_Brow']     = abs(m['L_Brow_Elev'] - m['R_Brow_Elev'])
        m['Asym_Mouth_Cnr']= abs(m['L_Cnr_Drop'] - m['R_Cnr_Drop'])
        m['Asym_Tilt']     = abs(m['L_Canthal_Tilt'] - m['R_Canthal_Tilt'])
        
        return m

#statisticalanalyzer

class ClinicalAuditor:
    def __init__(self):
        self.baseline_stats = None 
        
    def calibrate(self, df_history):
        if df_history.empty: return None
        stats = {}
        for col in df_history.select_dtypes(include=[np.number]).columns:
            mean = df_history[col].mean()
            std = df_history[col].std()
            if std == 0: std = 0.0001
            stats[col] = {'mean': mean, 'std': std}
        self.baseline_stats = stats
        return stats

    def generate_full_report(self, current):
        if not self.baseline_stats:
            return "NO BASELINE ESTABLISHED.", "GRAY"
            
        lines = []
        lines.append(f"CAPTURE DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*45)
        
        alert_level = "GREEN"
        significant_changes = 0
        
        categories = {
            "EYE MORPHOLOGY": ['L_Eye_H', 'R_Eye_H', 'L_Canthal_Tilt', 'R_Canthal_Tilt'],
            "BROW DYNAMICS": ['L_Brow_Elev', 'R_Brow_Elev'],
            "MOUTH & JAW": ['L_Cnr_Drop', 'R_Cnr_Drop', 'Mouth_Slant'],
            "EARS / WIDTH": ['L_Ear_Area', 'R_Ear_Area']
        }
        
        for cat, keys in categories.items():
            lines.append(f"\n[{cat}]")
            for k in keys:
                if k not in current or k not in self.baseline_stats: continue
                
                val = current[k]
                base = self.baseline_stats[k]
                delta = val - base['mean']
                z = delta / base['std']
                
                marker = " "
                if abs(z) > 2.0:
                    marker = "(!)"
                    significant_changes += 1
                elif abs(z) > 1.0:
                    marker = "(~)"
                
                unit = "°" if "Tilt" in k or "Slant" in k else ""
                lines.append(f" {marker} {k:<14} | Now: {val:.3f}{unit} | Base: {base['mean']:.3f}{unit} | Δ: {delta:+.3f}")

        lines.append("="*45)
        if significant_changes > 0:
            lines.append(f"RESULT: {significant_changes} SIGNIFICANT DEVIATIONS DETECTED.")
            lines.append("Review marked (!) items above.")
            alert_level = "RED"
        else:
            lines.append("RESULT: STABLE. No significant deviations from baseline.")
            
        return "\n".join(lines), alert_level

#revised professional GUI

class SymmetryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{Config.APP_NAME} // {Config.SUBTITLE}")
        self.root.geometry("1450x900")
        self.root.configure(bg=Config.THEME_BG)
        
        self.engine = FacialEngine()
        self.auditor = ClinicalAuditor()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, Config.CAM_WIDTH)
        self.cap.set(4, Config.CAM_HEIGHT)
        
        self.is_running = True
        self.can_capture = False
        self.current_mesh = None
        self.current_metrics = None
        self.mode = "MONITOR"
        
        # --- GHOST CONFIGURATION ---
        self.ghost_offsets = {
            'EYES':  {'x': 0, 'y': 0},
            'NOSE':  {'x': 0, 'y': 0},
            'MOUTH': {'x': 0, 'y': 0},
            'CHIN':  {'x': 0, 'y': 0},
            'EARS':  {'x': 0, 'y': 0},
            'GLOBAL': {'scale': 1.0, 'x': 0, 'y': 0}
        }
        
        # Calibration State
        self.active_feature_idx = 0
        self.feature_list = ['GLOBAL', 'EYES', 'NOSE', 'MOUTH', 'CHIN', 'EARS']
        
        self._load_data()
        self._build_layout()
        self._bind_controls() 
        self._update_loop()
        
        # FIX: Force focus to root so keybinds work immediately
        self.root.focus_set()
        
    def _load_data(self):
        if os.path.exists(Config.CSV_PATH):
            self.df = pd.read_csv(Config.CSV_PATH)
            baselines = self.df[self.df['session_type'] == 'BASELINE']
            if not baselines.empty:
                self.auditor.calibrate(baselines)
            else:
                self.mode = "BASELINE"
        else:
            self.df = pd.DataFrame()
            self.mode = "BASELINE"

    def _bind_controls(self):
        # Navigation (FIXED: return "break" prevents focus cycling)
        self.root.bind('<Tab>', self._cycle_feature)
        
        # Movement
        self.root.bind('<Left>', lambda e: self._adjust_ghost('x', -3))
        self.root.bind('<Right>', lambda e: self._adjust_ghost('x', 3))
        self.root.bind('<Up>', lambda e: self._adjust_ghost('y', -3))
        self.root.bind('<Down>', lambda e: self._adjust_ghost('y', 3))
        
        # Scaling
        self.root.bind('=', lambda e: self._adjust_scale(0.05))
        self.root.bind('-', lambda e: self._adjust_scale(-0.05))
        
        # Capture
        self.root.bind('<space>', self._capture)

    def _cycle_feature(self, event):
        self.active_feature_idx = (self.active_feature_idx + 1) % len(self.feature_list)
        feat = self.feature_list[self.active_feature_idx]
        self.lbl_calib.config(text=f"ADJUSTING: {feat}", fg=Config.THEME_ACTIVE)
        return "break" # CRITICAL FIX: Stops Tab from changing focus

    def _adjust_ghost(self, axis, delta):
        feat = self.feature_list[self.active_feature_idx]
        if feat == 'GLOBAL':
            self.ghost_offsets['GLOBAL'][axis] += delta
        else:
            self.ghost_offsets[feat][axis] += delta

    def _adjust_scale(self, delta):
        s = self.ghost_offsets['GLOBAL']['scale']
        self.ghost_offsets['GLOBAL']['scale'] = max(0.5, min(2.0, s + delta))

    def _build_layout(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Panel.TFrame", background=Config.THEME_PANEL, relief="flat")
        style.configure("H1.TLabel", font=("Helvetica", 20, "bold"), foreground=Config.THEME_ACCENT, background=Config.THEME_PANEL)
        
        self.root.columnconfigure(0, weight=1) 
        self.root.columnconfigure(1, weight=0, minsize=420) 
        self.root.rowconfigure(0, weight=1)
        
        # 1. VIDEO PANEL
        self.vid_panel = tk.Frame(self.root, bg="#000")
        self.vid_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_video = tk.Label(self.vid_panel, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        # Calibration Feedback
        self.lbl_calib = tk.Label(self.vid_panel, text="ADJUSTING: GLOBAL", bg="black", fg=Config.THEME_ACTIVE, font=("Arial", 16, "bold"))
        self.lbl_calib.place(relx=0.02, rely=0.02, anchor="nw")
        
        self.lbl_overlay = tk.Label(self.vid_panel, text="INITIALIZING...", bg="black", fg="white", font=("Arial", 24, "bold"))
        self.lbl_overlay.place(relx=0.5, rely=0.9, anchor="center")

        # 2. DATA PANEL
        data_panel = ttk.Frame(self.root, style="Panel.TFrame", width=420)
        data_panel.grid(row=0, column=1, sticky="ns", padx=(0,10), pady=10)
        data_panel.pack_propagate(False) 
        
        ttk.Label(data_panel, text=Config.APP_NAME, style="H1.TLabel").pack(pady=(20,5), anchor="w", padx=20)
        tk.Label(data_panel, text=Config.SUBTITLE, bg=Config.THEME_PANEL, fg="gray", font=("Arial", 9)).pack(pady=(0,20), anchor="w", padx=20)
        
        # Controls Card
        card_ctrl = tk.LabelFrame(data_panel, text="SESSION CONTROL", bg=Config.THEME_PANEL, fg="gray", font=("Arial", 9, "bold"))
        card_ctrl.pack(fill=tk.X, padx=15, pady=10)
        
        self.btn_mode = tk.Button(card_ctrl, text=f"CURRENT MODE: {self.mode}", 
                                  bg="#333", fg=Config.THEME_ACCENT, font=("Consolas", 11, "bold"),
                                  command=self._toggle_mode)
        self.btn_mode.pack(fill=tk.X, padx=10, pady=10)
        
        instr_frame = tk.Frame(card_ctrl, bg=Config.THEME_PANEL)
        instr_frame.pack(pady=5)
        tk.Label(instr_frame, text="[TAB] Cycle Features", bg=Config.THEME_PANEL, fg=Config.THEME_ACTIVE, font=("Arial", 9, "bold")).pack()
        tk.Label(instr_frame, text="[ARROWS] Move  |  [+/-] Scale Global", bg=Config.THEME_PANEL, fg="#bbb", font=("Arial", 8)).pack()

        # Live Bars
        card_bio = tk.LabelFrame(data_panel, text="SYMMETRY BALANCE (L vs R)", bg=Config.THEME_PANEL, fg="gray", font=("Arial", 9, "bold"))
        card_bio.pack(fill=tk.X, padx=15, pady=10)
        
        self.bars = {}
        for metric in ['Eye Height', 'Brow Elev', 'Mouth Cnr']:
            f = tk.Frame(card_bio, bg=Config.THEME_PANEL)
            f.pack(fill=tk.X, pady=5, padx=5)
            tk.Label(f, text=metric, width=12, anchor="w", bg=Config.THEME_PANEL, fg="white", font=("Consolas", 10)).pack(side=tk.LEFT)
            c = tk.Canvas(f, width=200, height=12, bg="#222", highlightthickness=0)
            c.pack(side=tk.RIGHT)
            self.bars[metric] = c

        # Log
        card_log = tk.LabelFrame(data_panel, text="STATUS STREAM", bg=Config.THEME_PANEL, fg="gray", font=("Arial", 9, "bold"))
        card_log.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.txt_log = tk.Text(card_log, bg="#111", fg="#0f0", font=("Courier New", 10), bd=0, padx=10, pady=10)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _toggle_mode(self):
        self.mode = "BASELINE" if self.mode == "MONITOR" else "MONITOR"
        color = Config.THEME_WARN if self.mode == "BASELINE" else Config.THEME_ACCENT
        self.btn_mode.config(text=f"CURRENT MODE: {self.mode}", fg=color)

    def _draw_hud(self, img, mesh, qc):
        h, w = img.shape[:2]
        overlay = img.copy()
        cx, cy = w // 2, h // 2
        
        # Global Transforms
        gs = self.ghost_offsets['GLOBAL']['scale']
        gx = cx + self.ghost_offsets['GLOBAL']['x']
        gy = cy + self.ghost_offsets['GLOBAL']['y']
        
        active_feat = self.feature_list[self.active_feature_idx]
        
        # Helper to draw components with independent offsets
        def draw_comp(tag, func, *args):
            # Base color
            col = (60, 60, 60)
            # Highlight active feature
            if active_feat == tag or active_feat == 'GLOBAL':
                col = (255, 255, 0) # Yellow highlight
                
            # Retrieve granular offset
            off_x = self.ghost_offsets[tag]['x']
            off_y = self.ghost_offsets[tag]['y']
            
            # Apply Global + Local
            final_cx = int(gx + off_x)
            final_cy = int(gy + off_y)
            
            func(final_cx, final_cy, col, *args)

        # Drawing Primitives
        def g_ellipse(cx, cy, col, ox, oy, rx, ry):
            cv2.ellipse(overlay, (int(cx + ox*gs), int(cy + oy*gs)), (int(rx*gs), int(ry*gs)), 0, 0, 360, col, 2)
            
        def g_line(cx, cy, col, x1, y1, x2, y2):
            cv2.line(overlay, (int(cx + x1*gs), int(cy + y1*gs)), (int(cx + x2*gs), int(cy + y2*gs)), col, 2)
        
        def g_circle(cx, cy, col, ox, oy, r):
            cv2.circle(overlay, (int(cx + ox*gs), int(cy + oy*gs)), int(r*gs), col, 2)

        # --- RENDER GHOST ---
        
        # Eyes
        draw_comp('EYES', g_ellipse, -80, -50, 45, 28)
        draw_comp('EYES', g_ellipse, 80, -50, 45, 28)
        
        # Nose
        draw_comp('NOSE', g_line, 0, -50, 0, 20)
        draw_comp('NOSE', g_circle, 0, 35, 15)
        
        # Mouth
        draw_comp('MOUTH', g_ellipse, 0, 85, 40, 20)
        
        # Chin
        draw_comp('CHIN', g_line, -30, 140, 30, 140)
        
        # Ears (Vertical Bars)
        draw_comp('EARS', g_line, -120, 20, -120, 80)
        draw_comp('EARS', g_line, 120, 20, 120, 80)

        # Radar
        scan_y = int((datetime.now().microsecond / 1000000) * h)
        cv2.line(overlay, (0, scan_y), (w, scan_y), (0, 50, 0), 1)
        
        # --- LIVE MESH ---
        color = (0, 255, 0) if qc['status'] == "OK" else (0, 0, 255)
        if mesh is not None:
            for poly in [Config.L_EYE_IDX, Config.R_EYE_IDX, Config.L_BROW, Config.R_BROW]:
                pts = mesh[poly][:, :2].astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, color, 1)
            
            if self.mode == "MONITOR":
                l_ear = mesh[Config.L_EAR_IDX][:, :2].astype(int).reshape((-1,1,2))
                r_ear = mesh[Config.R_EAR_IDX][:, :2].astype(int).reshape((-1,1,2))
                cv2.fillPoly(overlay, [l_ear], (0, 30, 0))
                cv2.fillPoly(overlay, [r_ear], (0, 30, 0))

        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    def _update_loop(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            mesh, qc = self.engine.process(frame)
            hud = self._draw_hud(frame, mesh, qc)
            
            if mesh is not None:
                self.current_mesh = mesh
                self.current_metrics = self.engine.extract_metrics(mesh)
                
                errs = []
                if abs(qc['yaw']) > Config.MAX_YAW: errs.append("CENTER FACE")
                if abs(qc['roll']) > Config.MAX_ROLL: errs.append("LEVEL HEAD")
                
                if errs:
                    self.can_capture = False
                    self.lbl_overlay.config(text=" | ".join(errs), fg="red")
                else:
                    self.can_capture = True
                    col = Config.THEME_ACCENT if self.mode == "MONITOR" else Config.THEME_WARN
                    self.lbl_overlay.config(text=f"READY ({self.mode})", fg=col)
                
                self._update_bars(self.current_metrics)
            else:
                self.lbl_overlay.config(text="NO SUBJECT", fg="gray")
                self.can_capture = False

            img = Image.fromarray(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
            
        self.root.after(30, self._update_loop)

    def _update_bars(self, m):
        pairs = [
            ('Eye Height', m['L_Eye_H'], m['R_Eye_H']),
            ('Brow Elev',  m['L_Brow_Elev'], m['R_Brow_Elev']),
            ('Mouth Cnr',  m['L_Cnr_Drop'], m['R_Cnr_Drop'])
        ]
        
        for name, l_val, r_val in pairs:
            canv = self.bars[name]
            canv.delete("all")
            w = 200 
            h = 12
            mid = w / 2
            diff = l_val - r_val
            scale = diff * 800 
            
            canv.create_line(mid, 0, mid, h, fill="#555")
            col = "#00ff00" if abs(diff) < 0.02 else "#ffaa00"
            if scale > 0:
                canv.create_rectangle(mid, 2, mid + scale, 10, fill=col, outline="")
            else:
                canv.create_rectangle(mid + scale, 2, mid, 10, fill=col, outline="")
                
    def _show_report_popup(self, text, alert_color):
        popup = tk.Toplevel(self.root)
        popup.title("CLINICAL ANALYSIS REPORT")
        popup.geometry("600x700")
        popup.configure(bg="#222")
        bg_col = Config.THEME_ALERT if alert_color == "RED" else "#228822"
        tk.Label(popup, text="DETAILED METRIC ANALYSIS", bg=bg_col, fg="white", 
                 font=("Helvetica", 14, "bold"), pady=15).pack(fill=tk.X)
        txt = scrolledtext.ScrolledText(popup, bg="#111", fg="white", font=("Consolas", 11), padx=15, pady=15)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, text)
        txt.config(state=tk.DISABLED)
        tk.Button(popup, text="CLOSE REPORT", command=popup.destroy, 
                  bg="#444", fg="white", font=("Arial", 10, "bold"), pady=10).pack(fill=tk.X)

    def _capture(self, event):
        if not self.can_capture: return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = self.current_metrics.copy()
        record['timestamp'] = ts
        record['session_type'] = self.mode
        
        df = pd.DataFrame([record])
        if not os.path.exists(Config.CSV_PATH):
            df.to_csv(Config.CSV_PATH, index=False)
        else:
            df.to_csv(Config.CSV_PATH, mode='a', header=False, index=False)
            
        self.txt_log.insert(tk.END, f"\n[{ts[-8:]}] CAPTURE: {self.mode}")
        self.txt_log.see(tk.END)
        
        if self.mode == "BASELINE":
            messagebox.showinfo("BASELINE SET", "Baseline metrics captured.\nSystem switching to MONITOR mode.")
            self._load_data()
            self.mode = "MONITOR"
            self.btn_mode.config(text=f"CURRENT MODE: {self.mode}", fg=Config.THEME_ACCENT)
        elif self.mode == "MONITOR":
            report_text, alert_lvl = self.auditor.generate_full_report(self.current_metrics)
            self._show_report_popup(report_text, alert_lvl)
        
        self.lbl_overlay.config(text="CAPTURED", fg="white")
        self.root.after(500, lambda: self.lbl_overlay.config(text=""))

    def on_close(self):
        self.is_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SymmetryGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()