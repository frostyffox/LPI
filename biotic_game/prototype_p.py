import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import tkinter as tk


#_______________SET UP_________________

#authomatically detect the screen dimension 
#open the video and the plot next to each other
root = tk.Tk() 
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()


#______Video Source______
source_path = 'euglena.mov'
#cap = cv2.VideoCapture(source_path)  #if i use a video saved on pc
cap = cv2.VideoCapture(1)  #if i plug my phone with usb 

#____Tracking parameters______
maxTrail = 30 
min_area , max_area = 100, 1500
dist_threshold = 40 #distance between different organisms
blur_kernel = (11,11)
light_position = np.array((320,0)) #place the light source at top center
max_euglena = 30 #i will use only the first 30 organisms detected 


path = "microorganism_trajectories.csv" #output path for csv 


#_______DATA STRUCTURES______
trails = {}
colors = {}
ojb_id = 0
trajectory_data = []


#_________MATPLOTLIB PLOT SET UP_______
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8,8)) #make sure the plot display is large enough
ax.set_facecolor("#0f0f0f")
ax.grid(color="#444", linestyle="--", linewidth=0.5)
ax.set_title("Euglena Trajectories", color="#00FFAA", fontname="monospace")
#ax.set_xlim(0, 640)
#ax.set_ylim(480, 0)

#id = 0 #object id
#tracks = {}

#______Window positioning______
mtpl = plt.get_current_fig_manager()
try:
    mtpl.window.wm_geometry(f"+{screen_w//2}+0")
except Exception:
    pass #backend fail

cv2.namedWindow("Microorganism detection", cv2.WINDOW_NORMAL) 
cv2.moveWindow("Microorganism detection", 0, 0)

plt.ion()



#________MAIN LOOP________
while True:
    ob, frame = cap.read() #grab next frame
    if not ob:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame_gray,blur_kernel, 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,
        cv2.THRESH_BINARY_INV, 15, 3)
    

    #_____DETECTING EUGLENA________
    #look for contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cont in contours:
        area = cv2.contourArea(cont)

        #filter contours found for size
        if min_area < area < max_area:
            (x, y, w, h) = cv2.boundingRect(cont)
            cc = (int(x+ w/2), int(y + h/2)) #center
            centers.append(cc)

            cv2.circle(frame, cc, 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 255), 1)



    #________TRACKING_________
    for cc in centers:
        found = False
        for tid, pps in trails.items():
            if np.linalg.norm(np.array(cc) - np.array(pps[-1])) < dist_threshold:
                pps.append(cc)
                found  = True
                break
        if not found and len(trails) < max_euglena:
            ojb_id += 1
            trails[ojb_id] = deque(maxlen = maxTrail)
            trails[ojb_id].append(cc)
            colors[ojb_id] = tuple(np.random.randint(0,255,3).tolist())
    
    #__________PLOT UPDATE_________
    ax.clear()
    ax.set_facecolor("#0f0f0f")
    ax.grid(color="#444", linestyle= "--", linewidth=0.5)
    ax.set_title("Average direction", color="#00FFAA",fontname="monospace")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.axhline(0, color='#555', linewidth=0.5)
    ax.axvline(0, color='#555', linewidth=0.5)

    all_vec = []

    for tid, pps in trails.items():
        if len(pps) > 1:
            start,end = np.array(pps[0]), np.array(pps[-1])
            all_vec.append(end - start)
            direct_to_target= np.linalg.norm(light_position - end) - np.linalg.norm(light_position - start) #direction to light
            trajectory_data.append({
                   "id": tid,
                    "start_x": start[0],"start_y": start[1],
                    "end_x": end[0], "end_y": end[1],
                    "towards_light": direct_to_target < 0
                    })
        
    if all_vec:
        avg_vector = np.mean(all_vec, axis =0)
        ax.arrow(0,0,avg_vector[0], -avg_vector[1],
                 head_width=10, head_length=15,
                 fc='#FF0066', ec='#FF0066')
        
        traj_ax = inset_axes(ax, width="35%", height = "35%", loc='lower right')
        traj_ax.set_facecolor("white")
        traj_ax.set_xlim(0, frame.shape[1])
        traj_ax.set_ylim(frame.shape[0], 0)
        traj_ax.set_title("Trajectories", fontsize=8, color="black")
        traj_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom = False)
        for tid, pps in trails.items():
            if len(pps)>1 and tid <= max_euglena:
                xs, ys = zip(*pps)
                traj_ax.plot(xs, ys, color=np.array(colors[tid])/255.0,linewidth=1)

    plt.pause(0.001)

    #_______VIDEO DISPLAY_______
    h, w = frame.shape[:2]
    scale = screen_h / h
    new_wid = int(w*scale)
    resize_frame = cv2.resize(frame, (new_wid, screen_h))

    cv2.imshow("Microorganism Detection", resize_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#_______SAVE AND EXPORT IN CsV______
df = pd.DataFrame(trajectory_data)
df.to_csv(path, index = False)
print(f"Trajectory data saved to {path}")

df = pd.read_csv(path)

#average astart and end of trajectories
avg_start = np.array([df['start_x'].mean(), df['start_y'].mean()])
avg_end = np.array([df['end_x'].mean(), df['end_y'].mean()])

#average trajecoty
average = avg_end - avg_start
print("Average movement:", average)

#light fixed at (320,0)
light_coord = np.array([320,0])
distance_start = np.linalg.norm(light_coord -  avg_start)
distance_end = np.linalg.norm(light_coord -  avg_end)

if distance_end < distance_start:
    print("Average movement is towards the light source :) ")
else: 
    print("Average movement is NOT towards the light source :( )")