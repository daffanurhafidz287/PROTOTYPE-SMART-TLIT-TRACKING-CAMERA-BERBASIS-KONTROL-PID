import cv2 
import numpy as np 
import serial 
import time 
 
# KONFIGURASI 
ARDUINO_PORT = 'COM12' 
BAUD_RATE = 115200  
TIMEOUT = 0.1 
 
# RANGE WARNA BIRU & OREN 
LOWER_BLUE = np.array([100, 150, 0]) 
UPPER_BLUE = np.array([140, 255, 255]) 
 
LOWER_ORANGE = np.array([5, 120, 120]) 
UPPER_ORANGE = np.array([20, 255, 255]) 
 
KERNEL = np.ones((5, 5), np.uint8) 
MIN_AREA = 1000  
 
DEAD_ZONE = 40   
SMOOTHING_FACTOR = 0.6   
MIN_MOVE_THRESHOLD = 1   
 
class PID: 
    def __init__(self, kp, ki, kd): 
        self.kp = kp 
        self.ki = ki 
        self.kd = kd 
        self.prev_error = 0 
        self.integral = 0 
        self.max_integral = 50   
 
    def compute(self, error): 
        if abs(error) < DEAD_ZONE: 
            self.integral = 0   
            self.prev_error = error 
            return 0 
             
        p = error * self.kp 
        self.integral += error 
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral)) 
        i = self.integral * self.ki 
        d = (error - self.prev_error) * self.kd 
        self.prev_error = error 
        return p + i + d 
 
# TUNING PID  
pid_x = PID(kp=0.015, ki=0.0, kd=0.008) 
pid_y = PID(kp=0.015, ki=0.0, kd=0.008) 
 
# Posisi servo dan target  
curr_pos_x = 90.0 
curr_pos_y = 90.0 
smooth_target_x = 90.0 
smooth_target_y = 90.0 
prev_sent_x = 90 
prev_sent_y = 90 
 
def initialize_serial(): 
    try: 
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=TIMEOUT) 
        time.sleep(2) 
        print(f"TERHUBUNG ke {ARDUINO_PORT}") 
        ser.write(b"90,90\n")  
        return ser 
    except Exception as e: 
        print(f"Error Serial: {e}") 
        return None 
 
def send_pid_command(ser, x_angle, y_angle): 
    global prev_sent_x, prev_sent_y 
    if not ser: return 
    if abs(x_angle - prev_sent_x) < MIN_MOVE_THRESHOLD and \ 
       abs(y_angle - prev_sent_y) < MIN_MOVE_THRESHOLD: 
    return 
     
    command = f"{int(x_angle)},{int(y_angle)}\n"  
    ser.write(command.encode('utf-8')) 
    prev_sent_x = int(x_angle) 
    prev_sent_y = int(y_angle) 
 
def get_color_at_point(hsv, x, y): 
    """Ambil nilai HSV di koordinat tertentu""" 
    if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]: 
        return hsv[y, x] 
    return None 
 
def main(): 
    global curr_pos_x, curr_pos_y, smooth_target_x, smooth_target_y 
    ser = initialize_serial() 
    cap = cv2.VideoCapture(1)  
 
    if not cap.isOpened(): 
        print("Kamera error!") 
        if ser: ser.close() 
        return 
 
    curr_pos_x = 90.0 
    curr_pos_y = 90.0 
    smooth_target_x = 90.0 
    smooth_target_y = 90.0 
    print("Tracking smooth dimulai. Tekan 'q' untuk keluar.") 
 
    while True: 
        ret, frame = cap.read() 
        if not ret: break 
         
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape 
        center_x, center_y = w // 2, h // 2 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
         
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE) 
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, KERNEL) 
         
        mask_orange = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE) 
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, KERNEL) 
         
        cnts_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts_o, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE) 
        cx_b, cy_b, cx_o, cy_o = None, None, None, None 
 
        # DETEKSI BIRU 
        if len(cnts_b) > 0: 
            c = max(cnts_b, key=cv2.contourArea) 
            if cv2.contourArea(c) > MIN_AREA:
                x, y, wb, hb = cv2.boundingRect(c) 
                cx_b, cy_b = x + wb//2, y + hb//2 
                cv2.rectangle(frame, (x, y), (x+wb, y+hb), (255, 0, 0), 2) 
                 
                # Ambil nilai HSV di titik tengah objek biru 
                hsv_val = get_color_at_point(hsv, cx_b, cy_b) 
                if hsv_val is not None: 
                    cv2.putText(frame, f"BIRU HSV: {hsv_val}", (x, y-10),  
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
 
        # DETEKSI OREN 
        if len(cnts_o) > 0: 
            c = max(cnts_o, key=cv2.contourArea) 
            if cv2.contourArea(c) > MIN_AREA: 
                x, y, wo, ho = cv2.boundingRect(c) 
                cx_o, cy_o = x + wo//2, y + ho//2 
                cv2.rectangle(frame, (x, y), (x+wo, y+ho), (0, 165, 255), 2) 
                # Ambil nilai HSV di titik tengah objek orange 
                hsv_val = get_color_at_point(hsv, cx_o, cy_o) 
                if hsv_val is not None: 
                    cv2.putText(frame, f"ORANGE HSV: {hsv_val}", (x, y-10),  
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
 
        # HITUNG MIDPOINT DAN PID 
        if cx_b is not None and cx_o is not None: 
 
            raw_midpoint_x = (cx_b + cx_o) / 2.0 
            raw_midpoint_y = (cy_b + cy_o) / 2.0 
            smooth_target_x = (SMOOTHING_FACTOR * raw_midpoint_x +  
                              (1 - SMOOTHING_FACTOR) * smooth_target_x) 
            smooth_target_y = (SMOOTHING_FACTOR * raw_midpoint_y +  
                              (1 - SMOOTHING_FACTOR) * smooth_target_y) 
            midpoint_x = int(smooth_target_x) 
            midpoint_y = int(smooth_target_y) 
             
            # Gambar Dead Zone 
            cv2.rectangle(frame, (center_x - DEAD_ZONE, center_y - DEAD_ZONE),  
                          (center_x + DEAD_ZONE, center_y + DEAD_ZONE), (255, 255, 255), 2) 
             
            # Garis penghubung dan Midpoint 
            cv2.line(frame, (cx_b, cy_b), (cx_o, cy_o), (0, 255, 255), 2) 
            cv2.circle(frame, (midpoint_x, midpoint_y), 8, (0, 0, 255), -1) 
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
            error_x = center_x - midpoint_x  
            error_y = center_y - midpoint_y 
 
            output_x = pid_x.compute(error_x) 
            output_y = pid_y.compute(error_y) 
 
            curr_pos_x -= output_x * 0.7   
            curr_pos_y -= output_y * 0.7 
 
            curr_pos_x = max(0, min(180, curr_pos_x)) 
            curr_pos_y = max(0, min(180, curr_pos_y)) 
 
            send_pid_command(ser, curr_pos_x, curr_pos_y) 
             
            # Status 
            is_static = abs(error_x) < DEAD_ZONE and abs(error_y) < DEAD_ZONE 
            action = "LOCKED" if is_static else "TRACKING" 
            color = (0, 255, 0) if is_static else (255, 200, 0) 
            cv2.putText(frame, f"{action}", (10, 30),  
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) 
            cv2.putText(frame, f"Servo: X={int(curr_pos_x)} Y={int(curr_pos_y)}", (10, 60),  
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
            cv2.putText(frame, f"Error: X={int(error_x)} Y={int(error_y)}", (10, 90),  
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
 
        # Crosshair di tengah 
        cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2) 
        cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2) 
 
        cv2.imshow("Color Tracking", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): break 
 
    cap.release() 
    cv2.destroyAllWindows() 
    if ser: ser.close() 
    print("Program selesai.") 
 
if __name__ == "__main__": 
    main()
