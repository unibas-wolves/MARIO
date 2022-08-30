import tkinter as tk
from unicodedata import is_normalized
from PIL import ImageTk, Image, ImageFont, ImageDraw
from tkinter import ttk, Label, simpledialog, OptionMenu, Toplevel, filedialog, PhotoImage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import statistics
import webbrowser
import cv2
import shutil
from functools import partial
import json
import subprocess as sub
import sys
sys.path.insert(0, "../")
from Homography import Homography
from Preparation import Preparation
from ResourceManager import ResourceManager
from PoseEstimator import PoseEstimator
from ObjectTracker import ObjectTracker
from Game import Game
from GameAnalyzer import GameAnalyzer
from Singleton import Singleton
from FallDetector import FallDetector
#########INTERFACCIA GRAFICA######################


        


def gui():

        device = False
 
        ###FINESTRA 2
        def open_vista_analysis(vista):
            vista_analysis= Toplevel(vista)
            screen_width = vista_analysis.winfo_screenwidth()
            screen_height = vista_analysis.winfo_screenheight()
            # find the center point
            window_width = 750
            window_height = 570
            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)-40
            vista_analysis.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            vista_analysis.resizable(False, False)
            vista_analysis.title('MARIO') 
            titolo1 = ttk.Label(
            vista_analysis,
            text="MARIO-Analysis",
            justify="center",
            font=("-size", 25, "-weight", "bold", "-slant", "italic"),
            foreground="#0071c1"
            )
            ga = GameAnalyzer("./game_data.csv")
            
            goal0, goal1, shots0, shots1, shots_target0, shots_target1, pass0, pass1 = ga.stats()
            poss0, poss1 = ga.ball_possession()  
            titolo1.grid(row=0, column=0, pady=10)    
            game_data = pd.read_csv("./game_data.csv")
            id_list = list(set(game_data[ga.game_data["team"] != -1]["id"].tolist()))     
            
            def aggiorna_labelsNew(goal0, goal1, shots0, shots1, shots_target0, shots_target1, poss0, poss1, pass0, pass1):
                finestra_stats = Toplevel(vista_analysis)
                screen_width = vista_analysis.winfo_screenwidth()
                screen_height = vista_analysis.winfo_screenheight()
                window_width = 470
                window_height = 400
                center_x = int(screen_width/2 - window_width / 2)
                center_y = int(screen_height/2 - window_height / 2)-40
                finestra_stats.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
                finestra_stats.resizable(False, False)
                finestra_stats.title('ViSta-stats')
                fnt = ImageFont.truetype("gui/Champions-Bold.ttf", 20)
                fntTeam = ImageFont.truetype("gui/Champions-Bold.ttf", 25)
                champions_image = Image.open("gui/champions.jpg")
                edit_image = ImageDraw.Draw(champions_image)
                s = Singleton()
                edit_image.text((20,155), s.getHomeTeam(), ("white"), fntTeam)
                edit_image.text((300,155), s.getAwayTeam() , ("white"), fntTeam)
                edit_image.text((190,155), str(goal0), ("white"), fnt)
                edit_image.text((260,155), str(goal1), ("white"), fnt)
                edit_image.text((20,215), str(poss0)+"\n"+str(shots0)+"\n"+str(shots_target0)+"\n"+str(pass0), ("white"), fnt, spacing=10)
                edit_image.text((350,215), str(poss1)+"\n"+str(shots1)+"\n"+str(shots_target1)+"\n"+str(pass1), ("white"), fnt, spacing=10)
            
                champions_image.save("./stats.png")
                global stats_image
                stats_image = PhotoImage(file="./stats.png")
                label=Label(finestra_stats, image=stats_image)
                label.pack(pady=20)
                label.config(image=stats_image)
              
           
            #POSSESSION
            stats_button = ttk.Button(
                vista_analysis,
                text='CALCULATE STATS',
                style="Accent.TButton",
                command=lambda: [aggiorna_labelsNew(goal0, goal1, shots0, shots1, shots_target0, shots_target1, poss0, poss1, pass0, pass1)])           
            stats_button.grid(row=10, column=3, padx=20, pady=10, sticky="nsew")            
            #HEATMAP           
            menu_heatmap = tk.Menu()
            menu_heatmap.add_command(label="Robot 1", command=lambda: ga.heatmap(1))
            menu_heatmap.add_command(label="Robot 2", command=lambda: ga.heatmap(2))
            menu_heatmap.add_command(label="Robot 3", command=lambda: ga.heatmap(3))
            menu_heatmap.add_command(label="Robot 4", command=lambda: ga.heatmap(4))
            menu_heatmap.add_command(label="Robot 5", command=lambda: ga.heatmap(5))
            menu_heatmap.add_command(label="Robot 6", command=lambda: ga.heatmap(6))
            menu_heatmap.add_command(label="Robot 7", command=lambda: ga.heatmap(7))
            menu_heatmap.add_command(label="Robot 8", command=lambda: ga.heatmap(8))
            menu_heatmap.add_command(label="Robot 9", command=lambda: ga.heatmap(9))
            menu_heatmap.add_command(label="Robot 10", command=lambda: ga.heatmap(10))
            menu_heatmap.add_command(label="ball", command=lambda: ga.heatmap(11))           
            menubutton_heatmap = ttk.Menubutton(
                vista_analysis, text="HEATMAP", menu=menu_heatmap, direction="below"
            )
            menubutton_heatmap.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")     
            #TRACKMAP            
            menu_trackmap = tk.Menu()
            menu_trackmap.add_command(label="Robot 1", command=lambda: ga.trackmap(1))
            menu_trackmap.add_command(label="Robot 2", command=lambda: ga.trackmap(2))
            menu_trackmap.add_command(label="Robot 3", command=lambda: ga.trackmap(3))
            menu_trackmap.add_command(label="Robot 4", command=lambda: ga.trackmap(4))
            menu_trackmap.add_command(label="Robot 5", command=lambda: ga.trackmap(5))
            menu_trackmap.add_command(label="Robot 6", command=lambda: ga.trackmap(6))
            menu_trackmap.add_command(label="Robot 7", command=lambda: ga.trackmap(7))
            menu_trackmap.add_command(label="Robot 8", command=lambda: ga.trackmap(8))
            menu_trackmap.add_command(label="Robot 9", command=lambda: ga.trackmap(9))
            menu_trackmap.add_command(label="Robot 10", command=lambda: ga.trackmap(10))
            menu_trackmap.add_command(label="ball", command=lambda: ga.trackmap(11))          

            menubutton_trackmap = ttk.Menubutton(
                vista_analysis, text="TRACKMAP", menu=menu_trackmap, direction="below"
            )
            menubutton_trackmap.grid(row=2, column=1, padx=20, pady=20, sticky="nsew")        
            #SHOT-PASS MAP
            shotmap_button = ttk.Button(
                vista_analysis,
                text='PASS-SHOT MAP',
                style="Accent.TButton",
                command=lambda: ga.pass_shot_map())
            shotmap_button.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")

            illegal_defender_button = ttk.Button(
                vista_analysis,
                text='ILL.DEF.',
                style="Accent.TButton",
                command=lambda: ga.illegal_defender())
            illegal_defender_button.grid(row=2, column=3, padx=20, pady=20, sticky="nsew")


        def open_vista_tracking(vista):
            open_vista_tracking = Toplevel(vista)
            screen_width = open_vista_tracking.winfo_screenwidth()
            screen_height = open_vista_tracking.winfo_screenheight()
            window_width = 750
            window_height = 570
            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)-40
            open_vista_tracking.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            open_vista_tracking.resizable(False, False)
            open_vista_tracking.title('MARIO')
            titolo_calibration = ttk.Label(open_vista_tracking, text="MARIO-Tracking", justify="center", font=("-size", 25, "-weight", "bold", "-slant", "italic"), foreground="#0071c1")
            titolo_calibration.grid(row=0, column=0, pady=10)
            

            

            go_to_analysis_button = ttk.Button(
                open_vista_tracking,
                text='Go TO ANALYSIS',
                style="Accent.TButton",
                command=lambda: open_vista_analysis(open_vista_tracking))
                
            go_to_analysis_button.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")


        def open_vista_stats(vista):
            open_vista_stats = Toplevel(vista)
            screen_width = open_vista_stats.winfo_screenwidth()
            screen_height = open_vista_stats.winfo_screenheight()
            window_width = 750
            window_height = 570
            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)-40
            open_vista_stats.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            open_vista_stats.resizable(False, False)
            open_vista_stats.title('MARIO')
            titolo_stats = ttk.Label(open_vista_stats, text="MARIO-Stats", justify="center", font=("-size", 25, "-weight", "bold", "-slant", "italic"), foreground="#0071c1")
            titolo_stats.grid(row=0, column=0, pady=10)


        ###FINESTRA 1
        root0 = tk.Tk()
        screen_width = root0.winfo_screenwidth()
        screen_height = root0.winfo_screenheight()
        window_width = 650
        window_height = 450
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)-40
        root0.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        root0.resizable(False, False)
        root0.title('MARIO')
        root0.tk.call('source', 'azure.tcl')
        root0.tk.call("set_theme", "light")
        wolves = Image.open("unibas_wolves.png")
        spqr = Image.open("spqr_team.png")
        uni = Image.open("unibas.jpeg")

        titolo0 = ttk.Label(
            root0,
            text="MARIO",
            justify="left",
            font=("-size", 25, "-weight", "bold", "-slant", "italic"),
            foreground="#0071c1"
        )
        titolo0.grid(row=0, column=0, pady=10)

        unibas_wolves = ImageTk.PhotoImage(wolves)
        button_wolves = ttk.Button(root0, text="wolves", image=unibas_wolves, command=partial(webbrowser.open, "https://sites.google.com/unibas.it/wolves"))
        button_wolves.grid(row=2, column=3, padx=10, pady=10, sticky="nsew")
        #label1 = tk.Label(image=test1)       
        #label1.image = test1
        #label1.place(x=600, y=100)
        
        spqr_team = ImageTk.PhotoImage(spqr)
        button_spqr = ttk.Button(root0, text="spqr", image=spqr_team, command=partial(webbrowser.open, "http://spqr.diag.uniroma1.it/"))
        button_spqr.grid(row=4, column=3, padx=10, pady=10, sticky="nsew")
        #label2 = tk.Label(image=test2)       
        #label2.image = test2
        #label2.place(x=610, y=400)
        
        unibas = ImageTk.PhotoImage(uni)
        button_unibas = ttk.Button(root0, text="unibas", image=unibas, command=partial(webbrowser.open, "https://portale.unibas.it/site/home.html"))
        button_unibas.grid(row=3, column=3, padx=10, pady=10, sticky="nsew")
        #label3 = tk.Label(image=test3)       
        #label3.image = test3
        #label3.place(x=600, y=250)
        

        
        #test4 = ImageTk.PhotoImage(titolo)
        #label4 = tk.Label(image=test4)       
        #label4.image = test4
        #label4.place(x=20, y=15)


        #FILE CHOOSEr

        def openVideo():
            root0.filenameVideos = filedialog.askopenfilename(initialdir="../video", title="Select video", filetypes=(("mp4 files", "*.mp4"), ("avi files", "*.avi"), ("all files", "*.*")))
            print(root0.filenameVideos)
            cap = cv2.VideoCapture(root0.filenameVideos)
            s = Singleton()
            s.setVideoPath(root0.filenameVideos)
            while(True):
                ret,frame = cap.read()
                cv2.imwrite("initial_frame.jpg",frame)
                break
            cap.release()
            uni = Image.open("initial_frame.jpg")
            size = 350,350
            uni.thumbnail(size, Image.ANTIALIAS)
            test3 = ImageTk.PhotoImage(uni)
            label3 = tk.Label(image=test3)       
            label3.image = test3
            label3.grid(row=3, column=0, columnspan=3, rowspan=3, padx=20, pady=20, sticky="nsew")

            return root0.filenameVideos 
            



        def open_cfg() :
            root0.filenameCfg = filedialog.askopenfilename(initialdir="../extrinsic_parameters", title="Select parameters files", filetypes=(("py files", "*.py"), ("all files", "*.*")))
            print(root0.filenameCfg)
            return root0.filenameCfg


        def open_gc_team() :
            root0.filenameGc_team = filedialog.askopenfilename(initialdir="../", title="Select gc team file", filetypes=(("json files", "*.json"), ("all files", "*.*")))
            print(root0.filenameGc_team)
            s = Singleton()
            s.set_gc_team_path(root0.filenameGc_team)
            return root0.filenameGc_team
            
        def open_gc_game() :
            root0.filenameGc_game = filedialog.askopenfilename(initialdir="../", title="Select gc game file", filetypes=(("json files", "*.json"), ("all files", "*.*")))

            s = Singleton()
            s.set_gc_game_path(root0.filenameGc_game)
            f = open(s.get_gc_team_path())              
            data = json.load(f)
            teamNum = []
            associationTeam = {}
            for i in data:
                teamNum.append((i['teamNum']))
            f.close()
            teamNum = list( dict.fromkeys(teamNum) )
            f = open(s.get_gc_game_path())              
            data = json.load(f)
            homeTeam = (data[1]["kickingTeam"])
            teamNum.remove(homeTeam)
            awayTeam = teamNum[0]
            f.close()
            f = open("gc/teams.txt")
            for line in f.readlines():
                key = line[0:line.find("=")]
                team = line[line.find("=")+1:line.find(",")]
                associationTeam[int(key)] = team
            s.setHomeTeam(associationTeam[homeTeam])
            s.setAwayTeam(associationTeam[awayTeam])
            return root0.filenameGc_game



        video_button = ttk.Button(
            root0,
            text='CHOOSE VIDEO',
            style="Accent.TButton",
            command=lambda:[ openVideo(), switch_calib.config(state="enabled")])
        video_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        cfg_button = ttk.Button(
            root0,
            text='CHOOSE CFG',
            style="Accent.TButton",
            command=lambda: open_cfg())
        cfg_button.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        gc_button = ttk.Button(
            root0,
            text='GAME CONTROLLER',
            style="Accent.TButton",
            command=lambda: [open_gc_team(), open_gc_game()])
        gc_button.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")
        
        def openCalib():
            root0.fileCalib = filedialog.askopenfilename(initialdir="../calibration_data", title="Select file", filetypes=(("npz files", "*.npz"), ("all files", "*.*")))
            s = Singleton()
            s.setCalibPath(root0.fileCalib)

            return root0.fileCalib
        

        

        calib_button = ttk.Button(root0, text="CHOOSE CALIB", style="Accent.TButton", command= lambda: [openCalib(), start_calibration_button.config(state="enabled")])
        calib_button.grid(row=6, column=1, padx=20, pady=10, sticky="nsew")


        def start_calibration(already_calib):
            global device 

            s = Singleton()
            print(s.getVideoPath())
            prep = Preparation( True)
            shutil.copyfile(s.getVideoPath(), "./output.avi")
            rm_config = {
                "homography_src_path": "./data/image_BG/background.jpg",
                "homography_dst_path": None,
                "tracker_src_path"   : "./output.avi"
            }

            new_field = prep.is_new_field()
            if new_field: 
                rm_config["homography_dst_path"] = "./field_new.png"
                print("RM CONFIG:", rm_config)
            else:
                rm_config["homography_dst_path"] = "./field_old.png"
                print("RM CONFIG:", rm_config)
            rm = ResourceManager(rm_config)
            src = rm.get_homography_src()
            dst = rm.get_homography_dst()

            h = Homography(src, dst)
            ot = ObjectTracker(
                yolo_weights_path="./data/detectionCalciatori.pt", 
                strong_sort_weights="./data/osnet_x1_0_msmt17.pth", 
                homography_handler=h,
                tracking_src= "./output.avi",
                device=device
            )
            fd = FallDetector(
                weight_path="./data/tsstg-model.pth",
                device=device
            )
            pe = PoseEstimator(
                "./data/pose_estimation.pth",
                device=device
            )

            game = Game(
                resource_manager=rm, 
                homography=h, 
                tracker=ot, 
                pose_estimator=pe,
                fall_detector=fd
            )
            print("ora calibro")
            prep.calibrate_video(already_calib)
            if(already_calib):
                prep.image_subtraction()
                h._from_detection()
                open_vista_tracking(root0)
                cv2.waitKey(1000)
                game.loop()


        #calibration_file_path = openCalib()
        #video_path = openVideo()
        

        import os
        os.chdir("..")

        start_calibration_button = ttk.Button(
            root0,
            text='START CALIBRATION',
            style="Accent.TButton",
            state="disabled",
            command=lambda: [start_calibration(False),  go_tracking_button.config(state="enabled")])
        start_calibration_button.grid(row=5, column=3, padx=20, pady=10, sticky="nsew")

        go_tracking_button = ttk.Button(
            root0,
            text='GO TO TRACKING',
            style="Accent.TButton",
            state="disabled",
            command=lambda: [start_calibration(True)])
            #command=lambda: open_vista_analysis(root0))
        go_tracking_button.grid(row=6, column=3, padx=20, pady=10, sticky="nsew")
        
            
        global is_on 
        is_on = True

        def switch() :
            global is_on
            if is_on == True:
                start_calibration_button.config(state="disabled")
                go_tracking_button.config(state="enabled")
                calib_button.config(state="disabled")
                is_on = False
            else:
                start_calibration_button.config(state="disabled")
                go_tracking_button.config(state="disabled")
                calib_button.config(state="enabled")
                is_on = True


        switch_calib = ttk.Checkbutton(root0, text="calibrated", state="disabled", style="Switch.TCheckbutton", command= lambda: switch())
        switch_calib.grid(row=6, column=0, padx=20, pady=20, sticky="nsew")
        
        global is_on_gpu 
        is_on_gpu = True
        
        def switch_gpu_fun() :
            global is_on_gpu, device
            if is_on_gpu == True:
                print("GPU ACCESA")
                is_on_gpu = False
                device    = "cuda"
            else:
                print("GPU SPENTA")
                is_on_gpu = True   
                device    = "cpu"
                
        switch_gpu = ttk.Checkbutton(root0, text="GPU", style="Switch.TCheckbutton", command= lambda: switch_gpu_fun())
        switch_gpu.grid(row=6, column=2, padx=20, pady=20, sticky="nsew")

        ####PROGRESS BAR

        
        root0.mainloop()
        
        
        
        

############## FINE INTERFACCIA GRAFICA #################

if __name__ == "__main__":

    gui()
    
    

 
