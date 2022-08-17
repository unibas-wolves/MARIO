import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import statistics
import webbrowser
import cv2
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import image as mpimg
from functools import partial


class GameAnalyzer:

    def __init__(self, game_data_path):
        self.game_data = pd.read_csv(game_data_path)
        self.game_data.loc[self.game_data['id'] == 11,'team'] = -1
        self.game_data.to_csv('./game_data.csv', index=False)
        self.length = 1024
        self.width = 503

    def draw_pitch(self):
        fig=plt.figure(figsize=(10,6))
        ax=fig.add_subplot(1,1,1)
        ax.set_facecolor('w')
        ax.invert_yaxis()
        #corner del campo

        linecolor = "black"
        plt.plot([0,0],[0,self.width], color=linecolor)
        plt.plot([0,self.length],[self.width,self.width], color=linecolor)
        plt.plot([self.length,self.length],[self.width,0], color=linecolor)
        plt.plot([self.length,0],[0,0], color=linecolor)
        plt.plot([self.length/2,self.length/2],[0,self.width], color=linecolor)

        plt.plot([72,72],[340,160], color="black")
        plt.plot([0, 72], [340,340], color="black")
        plt.plot([0, 72], [160,160], color="black")
        
        plt.plot([0,0],[195, 310], color="red")
        
        plt.plot([1024,1024],[195, 310], color="red")
        
        plt.plot([951,951],[340,160], color="black")
        plt.plot([1024,951], [340,340], color="black")
        plt.plot([1024,951], [160,160], color="black")
        centreCircle = plt.Circle((512,250),80,color="black",fill=False)
        centreSpot = plt.Circle((512,250),2,color="black")
        ax.add_patch(centreCircle)
        ax.add_patch(centreSpot)



    def illegal_defender(self):
        team0 = self.game_data[(self.game_data["team"] == 0) & (self.game_data["x_pos"] > 0) & (self.game_data["x_pos"] < 72) & (self.game_data["y_pos"] < 340) & (self.game_data["y_pos"] > 160)]
        team1 = self.game_data[(self.game_data["team"] == 1) & (self.game_data["x_pos"] > 951) & (self.game_data["x_pos"] < 1024) & (self.game_data["y_pos"] < 340) & (self.game_data["y_pos"] > 160)]

        team0_illegal = team0.groupby(["time"]).size().reset_index(name='counts')
        team1_illegal = team1.groupby(["time"]).size().reset_index(name='counts')

        team0_YES_illegal = team0_illegal[team0_illegal["counts"] >= 3]
        team1_YES_illegal = team1_illegal[team1_illegal["counts"] >= 3]

        illegal_counter0 = 0
        illegal_counter1 = 0

        for time in team0_YES_illegal["time"]:
            if team0_YES_illegal[team0_YES_illegal["time"] == time-1].shape[0] == 0:
                draw_pitch()
                print("Illegal defender for team 0 at frame", time, "with", team0[team0["time"] == time].shape[0], "robots in area")
                plt.scatter(team0[team0["time"] == time]["x_pos"], team0[team0["time"] == time]["y_pos"], s=10)
                plt.show()
                illegal_counter0 += 1


        for time in team1_YES_illegal["time"]:
            if team1_YES_illegal[team1_YES_illegal["time"] == time-1].shape[0] == 0:
                draw_pitch()
                print("Illegal defender for team 1 at frame", time, "with", team1[team1["time"] == time].shape[0], "robots in area")
                plt.scatter(team1[team1["time"] == time]["x_pos"], team1[team1["time"] == time]["y_pos"], s=10)
                plt.show()
                illegal_counter1 += 1
       
        print("TOTAL Team 0 illegal defenders:", illegal_counter0)
        print("TOTAL Team 1 illegal defenders:", illegal_counter1)
       
        return illegal_counter0, illegal_counter1
    
    
    def heatmap(self, player_id=None):
        if player_id == None:
            player_ids = set(
                self.game_data[self.game_data["team"] != -1]["id"].tolist()
            )
            for player_id, idx in zip(player_ids, range(0, len(player_ids))):
                self.draw_pitch()

                player_data = self.game_data[self.game_data["id"] == player_id]
                if len(player_data) == 1: continue

                y, x = np.linspace(0, self.length, 48), np.linspace(0, self.width, 34)
                density, _, _ = np.histogram2d(
                    player_data["y_pos"], 
                    player_data["x_pos"], 
                    [x, y]
                )
                plt.imshow(
                    density, 
                    interpolation="spline36", 
                    alpha=1, 
                    cmap="Reds", 
                    vmin=0, 
                    vmax=np.max(density), 
                    extent=[0, self.length, 0, self.width]
                )

                plt.title(f"Robot with ID {player_id} - Action Heat Map", color="black", size=20)
                plt.savefig(f"./plots/heatmaps/heatmap_robot_{player_id}.png")
                plt.close()
        else:
            self.draw_pitch()

            player_data = self.game_data[self.game_data["id"] == player_id]
            if len(player_data) == 1: 
                raise Exception("Not enough data to plot heatmap!")

            y, x = np.linspace(0, self.length, 48), np.linspace(0, self.width, 34)
            density, _, _ = np.histogram2d(
                player_data["y_pos"], 
                player_data["x_pos"],
                [x, y]
            )
            plt.imshow(
                density, 
                interpolation="spline36", 
                alpha=1, 
                cmap="Reds", 
                vmin=0, 
                vmax=np.max(density), 
                extent=[0, self.length, 0, self.width]
            )

            if player_id == 11:
                plt.title(f"Ball - Action heat map", color="black", size=20)
            else:
                plt.title(f"Robot with ID {player_id} - Action heat map", color="black", size=20)
            plt.savefig("heatmap.jpg")
        while True:
            cv2.imshow("Robot with ID {player_id} - Action heatmap map", cv2.imread("heatmap.jpg"))
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()

    def trackmap(self, player_id=None):
        if player_id == None:
            player_ids = set(
                self.game_data[self.game_data["team"] != -1]["id"].tolist()
            )
            for player_id, idx in zip(player_ids, range(0, len(player_ids))):
                
                self.draw_pitch()

                player_data = self.game_data[self.game_data["id"] == player_id]
                if len(player_data) == 1: continue
                plt.scatter(player_data['x_pos'], player_data['y_pos'], s=1.5)
                plt.title(f"Robot with ID {player_id} - Action Track Map", color="black", size=20)
                plt.savefig(f"./plots/heatmaps/heatmap_robot_{player_id}.png")
                plt.close()
        else:
        
            self.draw_pitch()

            player_data = self.game_data[self.game_data["id"] == player_id]
            if len(player_data) == 1: 
                raise Exception("Not enough data to plot heatmap!")
            plt.scatter(player_data['x_pos'], player_data['y_pos'], s=1.5)
            if player_id == 11:
                plt.title(f"Ball - Action track map", color="black", size=20)
            else:
                plt.title(f"Robot with ID {player_id} - Action track map", color="black", size=20)
            plt.savefig("trackmap.jpg")
        while True:
            cv2.imshow("Robot with ID {player_id} - Action track map", cv2.imread("trackmap.jpg"))
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
    def euclidean_dist(self, df1, df2, cols=['x_coord','y_coord']):
        return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)
        
    def ball_possession(self):
        team0 = self.game_data[self.game_data['team'] == 0]
        team1 = self.game_data[self.game_data['team'] == 1]
        ball = self.game_data[self.game_data['id'] == 11]
        righe = ball.shape[0]
        poss0 = 0
        poss1 = 0
        notcount = 0
        for time in ball['time']:
            ball_tmp = ball[ball['time'] == time][['x_pos', 'y_pos', 'time']]
            team0_tmp = team0[team0['time'] == time][['x_pos', 'y_pos', 'time']]
            team1_tmp = team1[team1['time'] == time][['x_pos', 'y_pos', 'time']]
            #print(ball_tmp)
            #print(team0_tmp)
            #print(team1_tmp)
            dist0  = self.euclidean_dist(team0_tmp, ball_tmp.head(1), cols=['x_pos', 'y_pos'])
            #print(dist0)
            dist1  = self.euclidean_dist(team1_tmp, ball_tmp.head(1), cols=['x_pos', 'y_pos'])
            #print(dist1)
            if dist0.size > 0 and dist1.size > 0 :
                min_dist0 = min(dist0)
                min_dist1 = min(dist1)
                if min_dist0 < min_dist1 :
                    poss0 += 1
                if min_dist1 < min_dist0:
                    poss1 += 1
                #print("poss 0", poss0)
                #print("poss 1", poss1)
            else :
                notcount += 1

        poss0 = round(poss0*100/(righe-notcount))
        poss1 = round(poss1*100/(righe-notcount))
        return poss0, poss1
    
    def stats(self):
        self.draw_pitch()
        ball = self.game_data[self.game_data['team'] == -1]
        goal0 = 0
        goal1 = 0
        shots0 = 0
        shots1 = 0
        shots_target0 = 0
        shots_target1 = 0
        pass0 = 0
        pass1 = 0
        listDist = []
        time = ball['time'].min()
        #for time in ball['time']:
        while time <= ball['time'].max() :
            ball20 = ball[ball['time'] == time+30][['x_pos', 'y_pos', 'time']]
            ball1 = ball[ball['time'] == time+5][['x_pos', 'y_pos', 'time']]
            ball0 = ball[ball['time'] == time][['x_pos', 'y_pos', 'time']]
            ball_meno20 = ball[ball['time'] == time-30][['x_pos', 'y_pos', 'time']]
            if not ball1.empty and not ball0.empty and not ball20.empty and not ball_meno20.empty:
                x1 = ball1['x_pos'].iloc[0]
                y1 = ball1['y_pos'].iloc[0]
                point1 = [x1, y1]
                x0 = ball0['x_pos'].iloc[0]
                y0 = ball0['y_pos'].iloc[0]
                point0 = [x0, y0]
                x20 = ball20['x_pos'].iloc[0]
                y20 = ball20['y_pos'].iloc[0]
                point20 = [x20, y20]
                x_m_20 = ball_meno20['x_pos'].iloc[0]
                y_m_20 = ball_meno20['y_pos'].iloc[0]
                point_m_20 = [x_m_20, y_m_20]
                dist0  = self.euclidean_dist(ball20, ball_meno20, cols=['x_pos', 'y_pos'])
                listDist.append(dist0)
                #pass0
                if dist0 > 50 and dist0 < 70 and x20 > x_m_20 :       
                    plt.plot([x_m_20,x20],[y_m_20,y20], color="red", linestyle="dashed")
                    plt.scatter(x20, y20, s=10, color="red", marker="o")
                    pass0 += 1
                    time += 40
                #pass1
                if dist0 > 50 and dist0 < 70 and x20 < x_m_20 :       
                    plt.plot([x_m_20,x20],[y_m_20,y20], color="blue", linestyle="dashed")
                    plt.scatter(x20, y20, s=10, color="blue", marker="o")
                    pass1 += 1
                    time += 40
                    
                    #and il primo parantesi area di destra secondo fuori area di destra y0 punro iniziale y1 punto finale
                #shots0
                if ((y1 < 340 and y1 > 160 and x1 < 1024 and x1 > 951) and (y0 > 340 or y0 < 160 or x0 < 951)) :
                    plt.plot([x0-5,x1+10],[y0-5,y1-10], color="red")
                    plt.scatter(x1+10, y1-10, s=10, color="red", marker="o")
                    shots0 += 1
                    time += 10
                #shots1       
                if ((y1 < 340 and y1 > 160 and x1 < 72 and x1 > 0) and (y0 > 340 or y0 < 160 or x0 > 72)):
                    plt.plot([x0-5,x1+10],[y0+5,y1-10], color="blue")
                    plt.scatter(x1+10, y1-10, s=10, color="blue", marker="o")
                    shots1 += 1   
                    time += 10        
                    
                    #y comprese nella linea tra i due pali            
                #shots_target0
                if((y1 < 310 and y1 > 195 and x1 < 1024 and x1 > 1000) and (y0 > 340 or y0 < 160 or x0 < 951)) :
                    plt.plot([x0+5,x1+10],[y0+5,y1-10], color="red")
                    plt.scatter(x1+10, y1-10, s=10, color="red", marker="o")
                    shots_target0 += 1 
                    time += 10
                #shots_target0 
                if ((y1 < 310 and y1 > 195 and x1 < 20 and x1 > 0) and (y0 > 340 or y0 < 160 or x0 > 20)):
                    plt.plot([x0-5,x1+10],[y0+5,y1-10], color="blue")
                    plt.scatter(x1+10, y1-10, s=10, color="blue", marker="o")
                    shots_target1 += 1 
                    time += 10
                #goal0 y tra i due pali x1 maggiore della linea di porta x0 minore linea di porta
                if(y1 < 310 and y1 > 195 and x1 >= 1024 and x0 < 1020):
                    plt.scatter(x1, y1, s=10, color="red", marker="X")
                    goal0 += 1
                #goal1
                if(y1 < 310 and y1 > 195 and x1 < 0 and x0 > 4):
                    plt.scatter(x1, y1, s=10, color="blue", marker="X")
                    goal1 += 1
            time += 1  
        plt.title(f"pass map and shot map", color="black", size=20) 
        legend_elements = [Line2D([0], [0], color='r', lw=2, linestyle='dashdot', label='pass0'), Line2D([0], [0], color='b', linestyle='dashdot', lw=2, label='pass1'), Line2D([0], [0], color='r', lw=2, label='shot0'), Line2D([0], [0], color='b', lw=2, label='shot1'), Line2D([0], [0], marker='X', color='w', label='goal0',
        markerfacecolor='r', markersize=10), Line2D([0], [0], marker='X', color='w', label='goal1',
        markerfacecolor='b', markersize=10)]
        plt.legend(handles=legend_elements, loc='upper left') 
        plt.savefig(f"passmap.jpg")
        plt.close()
        return goal0, goal1, shots0, shots1, shots_target0, shots_target1, pass0, pass1

    def pass_shot_map(self):   
        while True:
            cv2.imshow("Pass shot map", cv2.imread("passmap.jpg"))
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
