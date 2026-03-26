import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import cv2 
import numpy as np
import pandas as pd

from skimage.morphology import skeletonize
from skimage.draw import line_nd

import matplotlib.pyplot as plt

import scipy.ndimage as ndi

import os

from skan import Skeleton, sholl_analysis, draw

import tifffile

import math



class Sholl_analysis():
    
    def __init__(self, image_path):

        self.image_path = os.fsdecode(os.path.abspath(image_path))
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        

        self.image_for_center = cv2.imread(self.image_path)
        
        

    def get_soma_center(self):
        
        self.image_center = self.image_for_center.mean(axis =-1).astype('int')

        self.cy, self.cx = ndi.center_of_mass(self.image_center)

        return(self.cy, self.cx)
    
    

    def image_binary(self):
        
        self._, self.binary_img = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return(self.binary_img)

    #make skeleton
    def skeleton(self):
        
        self.skeleton = skeletonize(self.image)
    
        
        return(self.skeleton)

    #Maximum limit to which the circles will be drawn 
    def get_max_radius(self):

        self.max_radius = int(max(self.image.shape) * 0.99)

        return(self.max_radius)
        
        
    def get_image(self):
        return(self.image)
    
    #returns image with circles
    def get_img_with_circles(self):
        self.img_with_circles = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        return(self.img_with_circles)

class Masters:

    def __init__(self):
    

        self.root = tk.Tk()
        self.root.geometry("500x380")

        self.image_path = ""
        self.file_destination = ""
        self.directory_path = ""
        self.radius = 0
        self.distance = 0
        
        

        self.frame = tk.Frame(self.root)
        self.frame.grid(sticky = "nsew")

        self.root.grid_columnconfigure(0, weight = 1)
        self.root.grid_rowconfigure(0, weight= 1)
        
        self.label = tk.Label(self.frame, text = "Sholls's analysis", font = ('Arial', 14))
        self.label.grid(row=0, column=0, padx=190, pady=10, sticky="w")

        self.ask_type_label = tk.Label(self.frame, text = 'Anlysis type:', font = ('Arial', 12))
        self.ask_type_label.grid(row=1, column = 0, padx = 150, pady = 10, sticky="w")

        self.ask_type = ttk.Combobox(self.frame, width = 15)
        self.ask_type['values'] = ("Standard Sholl's", 'Lines')
        self.ask_type.grid(row = 1, column = 0, padx= 250, pady = 10, sticky="w")
        self.ask_type.current(0)

        self.ask_distance = tk.Text(self.frame, height = 1, width = 6)
        self.ask_distance.grid(row= 2, column= 0, padx = 240, pady = 10, sticky= "w")

        self.ask_distance_label = tk.Label(self.frame, text = "Distance: ", font = ('Arial', 12))
        self.ask_distance_label.grid(row = 2, column = 0, padx = 160, pady = 10, sticky = "w")
        
        self.ask_units = ttk.Combobox(self.frame, width = 8)
        self.ask_units['values'] = ('pixels', 'microns')
        self.ask_units.grid(row = 2, column = 0, padx = 305, pady = 10, sticky = "w")
        self.ask_units.current(0)
        
        self.var1 = tk.IntVar()
        self.chkPicture = tk.Checkbutton(self.frame, text = "One picture", font = ('Arial', 12), variable = self.var1, command= self.get_var1)
        self.chkPicture.grid(row=3, column=0, pady = 10, padx = 120, sticky="w")

        self.var2 = tk.IntVar()
        self.chkPicture2 = tk.Checkbutton(self.frame, text = "Many pictures", font = ('Arial', 12), variable = self.var2, command = self.get_var2)
        self.chkPicture2.grid(row=3, column=0, pady = 10, padx = 250, sticky="w")

        
        self.get_file_path = tk.Button(self.frame, text = "Choose File/s", font = ('Arial', 10), command = self.choose_file)
        self.get_file_path.grid(row = 4, column = 0, pady = 10, padx = 205, sticky="w")

        self.target_file_path = tk.Button(self.frame, text= "Destination File", font = ('Arial', 10), command = self.destination_file)
        self.target_file_path.grid(row = 5, column = 0, pady = 10, padx = 200, sticky="w")

        self.ask_end_file_name = tk.Text(self.frame, height = 1, width = 10)
        self.ask_end_file_name.grid(row = 6, column = 0, pady = 10, padx = 250, sticky="w")

        self.ask_end_file_name_label = tk.Label(self.frame, text= "Excel file name: ", font = ('Arial', 10))
        self.ask_end_file_name_label.grid(row = 6, column = 0, padx = 145, pady = 10, sticky="w")

        self.run_analysis_btn = tk.Button(self.frame, text = "Run", font = ('Arial', 10), command=self.run_analysis)
        self.run_analysis_btn.grid(row = 7, column=0, pady = 10, padx = 235, sticky="w")

        self.root.mainloop()
    
    
    def make_plot_lines(self,compare_matrix, skeleton, line_matrix, cy, cx, distance, ratio_microns_per_pixels, ratio_pixels_per_microns):
        if (self.ask_units.get() == 'pixels'):
            self.columns = []
            self.data = []
            self.cy_neg = round(cy)
            self.cy_pos = round(cy)
            self.center = np.array([cy,cx])
            self.skeleton = Skeleton(skeleton)
            self.line_matrix = Skeleton(line_matrix)

            
            for i  in range(0,compare_matrix.shape[0], distance):
                
                if i==0:
                    self.columns.append(i)
                else:
                    self.columns.append(i)
                    self.columns.append(-i)

            for j in range(0, compare_matrix.shape[0]):
                
                if j ==0:
                    self.row_neg = compare_matrix[self.cy_neg,:]
                    self.data.append(self.row_neg.sum())
                    self.cy_neg += distance
                    self.cy_pos -= distance
                else:
                    try:
                        self.row_neg = compare_matrix[self.cy_neg,:]
                        self.row_pos = compare_matrix[self.cy_pos,:]
                        
                        self.data.append(self.row_pos.sum())
                        self.data.append(self.row_neg.sum())
                        

                        self.cy_neg += distance
                        self.cy_pos -= distance
                    except:
                        break

            
            for i in range(len(self.columns)-len(self.data)):
                self.data.append(np.int64(0))
            
            self.table = pd.DataFrame({'distance': self.columns, 'crossings': self.data}).sort_values(by = 'distance')
            self.start_index = 0
            self.end_index = 0

            for index in range(len(self.table['crossings'])):
                if self.table['crossings'].iloc[index] != 0:
                    self.start_index = index - 1
                    break
            for index in range(len(self.table['crossings'])):
                if (self.table['crossings'].iloc[index-1] != 0) and (self.table['crossings'].iloc[index] == 0):
                    self.end_index = index
                    break


            self.table = self.table.iloc[self.start_index-1:self.end_index+2,:]  
            
            self.fig, (self.ax0, self.ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
            
            
            draw.overlay_skeleton_2d_class(
                        self.skeleton, skeleton_colormap='viridis_r', vmin=0, axes=self.ax0
                        )
            draw.overlay_skeleton_2d_class(
                        self.line_matrix, skeleton_colormap='cool_r', vmin=0, axes=self.ax0
                        )
            
            
            self.ax0.autoscale_view()
            self.ax0.set_facecolor('black')
            self.ax0.set_aspect('equal')
            self.ax0.invert_yaxis()
            
            

            self.ax1.plot('distance', 'crossings', data=self.table, marker='o')
            self.ax1.set_xlabel('pixels')
            self.ax1.set_ylabel('crossings')
            
            
            plt.show()

        elif (self.ask_units.get() == 'microns'):

            self.columns = []
            self.data = []
            self.cy_neg = round(cy)
            self.cy_pos = round(cy)
            self.center = np.array([cy,cx])
            self.skeleton = Skeleton(skeleton)
            self.line_matrix = Skeleton(line_matrix)


            self.range = np.arange(0, compare_matrix.shape[0], distance*ratio_pixels_per_microns)
            
            for i in self.range:
                if i == 0:
                    self.columns.append(i)
                else:
                    self.columns.append(i)
                    self.columns.append(-i)
            

            for j in range(0, compare_matrix.shape[0]):
                
                if j ==0:
                    self.row_neg = compare_matrix[self.cy_neg,:]
                    self.data.append(self.row_neg.sum())
                    self.cy_neg -= int(round(distance * ratio_pixels_per_microns,0))
                    self.cy_pos += int(round(distance * ratio_pixels_per_microns,0))
                    
                else:
                    try:
                        self.row_neg = compare_matrix[self.cy_neg,:]
                        self.row_pos = compare_matrix[self.cy_pos,:]
                        
                        self.data.append(self.row_neg.sum())
                        self.data.append(self.row_pos.sum())
                        

                        self.cy_neg -= int(round(distance * ratio_pixels_per_microns,0))
                        self.cy_pos += int(round(distance * ratio_pixels_per_microns,0))
                        
                    except Exception as e:
                        break


            for i in range(len(self.columns)-len(self.data)):
                self.data.append(np.int64(0))
            
            
            self.table = pd.DataFrame({'distance': self.columns, 'crossings': self.data}).sort_values(by = 'distance')
            self.start_index = 0
            self.end_index = 0
            
            

            for index in range(len(self.table['crossings'])):
                if self.table['crossings'].iloc[index] != 0:
                    self.start_index = index - 1
                    break
            for index in range(len(self.table['crossings'])):
                if (self.table['crossings'].iloc[index-1] != 0) and (self.table['crossings'].iloc[index] == 0):
                    self.end_index = index
                    break


            self.table = self.table.iloc[self.start_index-1:self.end_index+2,:]
            self.table['distance'] = self.table['distance'].map(lambda x: x * ratio_microns_per_pixels)

              

            self.fig, (self.ax0, self.ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
            
            draw.overlay_skeleton_2d_class(
                        self.skeleton, skeleton_colormap='viridis_r', vmin=0, axes=self.ax0,
                        )
            draw.overlay_skeleton_2d_class(
                        self.line_matrix, skeleton_colormap='cool_r', vmin=0, axes=self.ax0
                        )
            
            
            self.ax0.autoscale_view()
            self.ax0.set_facecolor('black')
            self.ax0.set_ylim(0, skeleton.shape[0])
            self.ax0.set_xlim(0, skeleton.shape[0])
            self.ax0.set_aspect('equal')
            self.ax0.invert_yaxis()

            self.ax1.plot('distance', 'crossings', data=self.table, marker = 'o')
            self.ax1.set_xlabel('microns')
            self.ax1.set_ylabel('crossings')
            self.ax1.set_xticks(self.table['distance'])
            
            plt.show()
        
        else:
            pass


    def make_plot(self, skeleton, cx, cy, table, center, radii):
        
        if (self.ask_units.get() == 'pixels'):
            self.fig, (self.ax0, self.ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

            # draw the skeleton
            draw.overlay_skeleton_2d_class(
                    skeleton, skeleton_colormap='viridis_r', vmin=0, axes=self.ax0
                    )
            # draw the shells
            draw.sholl_shells(center, radii, axes=self.ax0)
            # fiddle with plot visual aspects
            self.ax0.autoscale_view()
            self.ax0.set_facecolor('black')
            self.ax0.set_ylim(0, cy * 2)
            self.ax0.set_xlim(0, cx * 2)
            self.ax0.set_aspect('equal')

            # in second subplot, plot the Sholl analysis
            self.ax1.plot('radius', 'crossings', data=table, marker = 'o')
            self.ax1.set_xlabel('radius (pixels)')
            self.ax1.set_ylabel('crossings')

            plt.show()
        elif (self.ask_units.get() == 'microns'):
            self.fig, (self.ax0, self.ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

            # draw the skeleton
            draw.overlay_skeleton_2d_class(
                    skeleton, skeleton_colormap='viridis_r', vmin=0, axes=self.ax0
                    )
            # draw the shells
            draw.sholl_shells(center, radii, axes=self.ax0)
            # fiddle with plot visual aspects
            self.ax0.autoscale_view()
            self.ax0.set_facecolor('black')
            self.ax0.set_ylim(0, cy * 2)
            self.ax0.set_xlim(0, cx * 2)
            self.ax0.set_aspect('equal')
            
            
            # in second subplot, plot the Sholl analysis
            self.ax1.plot('radius', 'crossings', data=table, marker = 'o')
            self.ax1.set_xlabel('radius (microns)')
            self.ax1.set_ylabel('crossings')
   
            
            plt.show()
        else:
            print('Error')



    def run_analysis_circles(self):
        #If user selected to check one picture
        if (self.var1.get() == 1): 
            #If units are in pixels
            if (self.ask_units.get() == 'pixels'):

                self.sholl = Sholl_analysis(self.image_path)
                self.skeleton = self.sholl.skeleton()
                self.skeleton = Skeleton(self.skeleton)
                
                self.cy, self.cx = self.sholl.get_soma_center()
                self.center = np.array([self.cy, self.cx])

                try:
                    self.radius = int(self.ask_distance.get("1.0", "end-1c"))
                except:
                    self.radius = float(self.ask_distance.get("1.0", "end-1c"))

                self.max_radius = self.sholl.get_max_radius()
                self.radii = np.arange(self.radius, self.max_radius, self.radius)


                self.center, self.radii, self.counts = sholl_analysis(self.skeleton, center = self.center, shells = self.radii)
                self.table = pd.DataFrame({'radius':self.radii, 'crossings':self.counts})
                

                self.make_plot(self.skeleton, self.cx, self.cy, self.table, self.center, self.radii)


             #If units are in microns
            elif (self.ask_units.get() == 'microns'):
                
        
                self.sholl = Sholl_analysis(self.image_path)
                self.skeleton = self.sholl.skeleton()
                self.skeleton = Skeleton(self.skeleton)

                self.cy, self.cx = self.sholl.get_soma_center()
                self.center = np.array([self.cy, self.cx])

                try:
                    self.radius = int(self.ask_distance.get("1.0", "end-1c"))
                except:
                    self.radius = float(self.ask_distance.get("1.0", "end-1c"))
                
                #Getting resolution in pixels and microns for the metadata of the picrue
                with tifffile.TiffFile(self.image_path) as tif:
                    self.tags = tif.pages[0].tags
                    self.resolution_x = self.tags.get('XResolution')
                    print(self.tags)

                    self.ratio_pixels_per_microns_x = self.resolution_x.value[0]/self.resolution_x.value[1]
                    self.ratio_microns_per_pixels_x = self.resolution_x.value[1]/self.resolution_x.value[0]

                    print(self.ratio_pixels_per_microns_x, self.ratio_microns_per_pixels_x)
                

                #Converting microns to pixels, Sholl's library works with pixels
                self.radius_pixels = self.ratio_pixels_per_microns_x * self.radius
                print(self.radius_pixels)

                self.max_radius = self.sholl.get_max_radius()
                self.radii = np.arange(self.radius_pixels, self.max_radius, self.radius_pixels)

                self.center, self.radii, self.counts = sholl_analysis(self.skeleton, center = self.center, shells = self.radii)

                #Creating table for plot and converting pixels to microns
                self.table = pd.DataFrame({'radius':self.radii, 'crossings':self.counts})
                self.table['radius'] = self.table['radius'].map(lambda x: x * self.ratio_microns_per_pixels_x)


                self.make_plot(self.skeleton, self.cx, self.cy, self.table, self.center, self.radii)
            
            else:
                pass
            
        #If user selected to test multiple pictures/folders
        elif self.var2.get() == 1:
            #If units are in pixels
            if (self.ask_units.get() == 'pixels'):
                #Making arrays for future dataframe
                self.columns = []
                self.data = []
                self.codes = []

                #Looping through file seleceted by user
                for file in os.listdir(self.directory_path):
                    try:
                        self.filepath = os.path.join(self.directory_path, file)
                        self.filename = os.fsdecode(file)
                        
                        
                        
                        self.sholl = Sholl_analysis(self.filepath)
                        self.skeleton = self.sholl.skeleton()
                        self.skeleton = Skeleton(self.skeleton)
                        
                        self.cy, self.cx = self.sholl.get_soma_center()
                        self.center = np.array([self.cy, self.cx])
                        try:
                            self.radius = int(self.ask_distance.get("1.0", "end-1c"))
                        except:
                            self.radius = float(self.ask_distance.get("1.0", "end-1c"))

                        self.max_radius = self.sholl.get_max_radius()
                        self.radii = np.arange(self.radius, self.max_radius, self.radius)
                        


                        self.center, self.radii, self.counts = sholl_analysis(self.skeleton, center = self.center, shells = self.radii)

                        if len(self.radii) > len(self.columns):
                            self.columns = self.radii
                        
                        self.data.append(self.counts)
                        self.codes.append(self.filename)
                    except:
                        continue
                    
                
                self.create_excel(self.data, self.codes, self.columns)

            #if units are in microns
            elif (self.ask_units.get() == 'microns'):

                #Making arrays for future dataframe
                self.columns = []
                self.data = []
                self.codes = []

                #Looping through file seleceted by user
                for file in os.listdir(self.directory_path):
                    try:
                        self.filepath = os.path.join(self.directory_path, file)
                        self.filename = os.fsdecode(file)
       
                        self.sholl = Sholl_analysis(self.filepath)
                        self.skeleton = self.sholl.skeleton()
                        self.skeleton = Skeleton(self.skeleton)
                        
                        self.cy, self.cx = self.sholl.get_soma_center()
                        self.center = np.array([self.cy, self.cx])

                        try:
                            self.radius = int(self.ask_distance.get("1.0", "end-1c"))
                        except:
                            self.radius = float(self.ask_distance.get("1.0", "end-1c"))

                        #Getting resolution in pixels and microns for the metadata of the picrue
                        with tifffile.TiffFile(self.filepath) as tif:
                            self.tags = tif.pages[0].tags
                            self.resolution = self.tags.get('XResolution')

                            self.ratio_pixels_per_microns = self.resolution.value[0]/self.resolution.value[1]
                            self.ratio_microns_per_pixels = self.resolution.value[1]/self.resolution.value[0]

                        
                        #Converting microns to pixels, Sholl's library works with pixels
                        self.radius_pixels = self.ratio_pixels_per_microns * self.radius
                        

                        self.max_radius = self.sholl.get_max_radius()
                        self.radii = np.arange(self.radius_pixels, self.max_radius, self.radius_pixels)
                        

                        self.center, self.radii, self.counts = sholl_analysis(self.skeleton, center = self.center, shells = self.radii)

                        if len(self.radii) > len(self.columns):
                            self.columns = self.radii
                        
                        self.data.append(self.counts)
                        self.codes.append(self.filename)
                    except:
                        continue
                      

                #Making columns, converting pixels back to microns
                self.columns = list(map(lambda x: x * self.ratio_microns_per_pixels, self.columns))
                
                self.create_excel(self.data, self.codes, self.columns)


    def run_analysis_lines(self):
        if (self.var1.get() == 1): 
            #If units are in pixels
            if (self.ask_units.get() == 'pixels'):

                self.sholl = Sholl_analysis(self.image_path)
                #self.skeleton = Skeleton(self.skeleton)
                self.skeleton = self.sholl.skeleton()

                self.cy, self.cx = self.sholl.get_soma_center()
                self.cy_neg = round(self.cy)
                self.cy_pos = round(self.cy)
                self.center = np.array([self.cy, self.cx])
                

                try:
                    self.distance = int(self.ask_distance.get("1.0", "end-1c"))
                except:
                    self.distance = float(self.ask_distance.get("1.0", "end-1c"))


                self.line_matrix = np.zeros((self.skeleton.shape[1],self.skeleton.shape[0]))


                self.max_distance = self.skeleton.shape[0]
                
                
                while self.cy_pos <= self.max_distance:
                    
                    try:
                        self.line_rightSide_up = line_nd((self.cy_pos, self.cx), (self.cy_pos,self.skeleton.shape[0]), endpoint = False)
                        self.line_leftSide_up = line_nd((self.cy_pos,self.cx), (self.cy_pos, 0), endpoint = False)
                        
                        self.line_matrix[self.line_rightSide_up] = 1
                        self.line_matrix[self.line_leftSide_up] = 1
                    
                        self.cy_pos += self.distance
                    except Exception as e:
                        break

                
                while self.cy_neg > 0:
                    
                    try:
                        
                        self.line_rightSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg,self.skeleton.shape[0]), endpoint = False)
                        self.line_leftSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg, 0), endpoint = False)
                        self.line_matrix[self.line_rightSide_down] = 1
                        self.line_matrix[self.line_leftSide_down] = 1

                        self.cy_neg -= self.distance
                    except Exception as e:
                        break  
                
                self.compare_matrix = np.where(np.logical_and(self.line_matrix == 1, self.skeleton == 1), 1, 0)
                

                for j in range(0,len(self.compare_matrix)):
                    self.cond = False
                    for i in range(0, len(self.compare_matrix[j])):
                        if self.compare_matrix[j][i] == 1 and self.cond == False:
                            self.cond = True
                        elif self.compare_matrix[j][i] == 1 and self.cond == True:
                            self.compare_matrix[j][i] = 0

                        elif self.compare_matrix[j][i] == 0 and self.cond == True:
                            self.cond = False
                        else:
                            pass
                
                self.make_plot_lines(self.compare_matrix, self.skeleton, self.line_matrix, self.cy, self.cx, self.distance, None, None)
                
                


            elif (self.ask_units.get() == 'microns'):
                self.sholl = Sholl_analysis(self.image_path)
                self.skeleton = self.sholl.skeleton()
                #self.skeleton = Skeleton(self.skeleton)
                
                self.cy, self.cx = self.sholl.get_soma_center()
                self.cy_neg = self.cy
                self.cy_pos = self.cy
                self.center = np.array([self.cy, self.cx])

                try:
                    self.distance = int(self.ask_distance.get("1.0", "end-1c"))
                except:
                    self.distance = float(self.ask_distance.get("1.0", "end-1c"))


                with tifffile.TiffFile(self.image_path) as tif:
                    self.tags = tif.pages[0].tags
                    self.resolution_x = self.tags.get('XResolution')
                    

                    self.ratio_pixels_per_microns_x = self.resolution_x.value[0]/self.resolution_x.value[1]
                    self.ratio_microns_per_pixels_x = self.resolution_x.value[1]/self.resolution_x.value[0]

                self.distance_in_pixels = int(round(self.ratio_pixels_per_microns_x * self.distance,0))

                print(f'p/m: {self.ratio_pixels_per_microns_x}, m/p: {self.ratio_microns_per_pixels_x}')


                self.line_matrix = np.zeros((self.skeleton.shape[0],self.skeleton.shape[1]))

                self.max_distance = self.skeleton.shape[0]

                while self.cy_pos <= self.max_distance:
                    
                    try:
                        self.line_rightSide_up = line_nd((self.cy_pos, self.cx), (self.cy_pos,self.skeleton.shape[0]), endpoint = False)
                        self.line_leftSide_up = line_nd((self.cy_pos,self.cx), (self.cy_pos, 0), endpoint = False)
                        
                        self.line_matrix[self.line_rightSide_up] = 1
                        self.line_matrix[self.line_leftSide_up] = 1
                    
                        self.cy_pos += self.distance_in_pixels
                    except Exception as e:
                        break

                
                while self.cy_neg > 0:
                    
                    try:
                        
                        self.line_rightSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg,self.skeleton.shape[0]), endpoint = False)
                        self.line_leftSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg, 0), endpoint = False)
                        self.line_matrix[self.line_rightSide_down] = 1
                        self.line_matrix[self.line_leftSide_down] = 1

                        self.cy_neg -= self.distance_in_pixels
                    except Exception as e:
                        break  
                    
                
                self.compare_matrix = np.where(np.logical_and(self.line_matrix == 1, self.skeleton == 1), 1, 0)
                

                for j in range(0,len(self.compare_matrix)):
                    self.cond = False
                    for i in range(0, len(self.compare_matrix[j])):
                        if self.compare_matrix[j][i] == 1 and self.cond == False:
                            self.cond = True
                        elif self.compare_matrix[j][i] == 1 and self.cond == True:
                            self.compare_matrix[j][i] = 0

                        elif self.compare_matrix[j][i] == 0 and self.cond == True:
                            self.cond = False
                        else:
                            pass
                
                self.make_plot_lines(self.compare_matrix, self.skeleton, self.line_matrix, self.cy, self.cx, self.distance, self.ratio_microns_per_pixels_x, self.ratio_pixels_per_microns_x)

        elif (self.var2.get() == 1):
            if (self.ask_units.get() == 'pixels'):
                
                self.all_data = []
                self.codes = []
                

                for file in os.listdir(self.directory_path):
                    try:
                        self.filepath = os.path.join(self.directory_path, file)
                        self.filename = os.fsdecode(file)

                        self.sholl = Sholl_analysis(self.filepath)
                        #self.skeleton = Skeleton(self.skeleton)
                        self.skeleton = self.sholl.skeleton()

                        self.cy, self.cx = self.sholl.get_soma_center()
                        self.cy_neg = round(self.cy)
                        self.cy_pos = round(self.cy)
                        self.center = np.array([self.cy, self.cx])

                        try:
                            self.distance = int(self.ask_distance.get("1.0", "end-1c"))
                        except:
                            self.distance = float(self.ask_distance.get("1.0", "end-1c"))
                        

                        self.line_matrix = np.zeros((self.skeleton.shape[1],self.skeleton.shape[0]))


                        self.max_distance = self.skeleton.shape[0]
                        
                        
                        while self.cy_pos <= self.max_distance:
                            
                            try:
                                self.line_rightSide_up = line_nd((self.cy_pos, self.cx), (self.cy_pos,self.skeleton.shape[0]), endpoint = False)
                                self.line_leftSide_up = line_nd((self.cy_pos,self.cx), (self.cy_pos, 0), endpoint = False)
                                
                                self.line_matrix[self.line_rightSide_up] = 1
                                self.line_matrix[self.line_leftSide_up] = 1
                            
                                self.cy_pos += self.distance
                            except Exception as e:
                                break

                        
                        while self.cy_neg > 0:
                            
                            try:
                                
                                self.line_rightSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg,self.skeleton.shape[0]), endpoint = False)
                                self.line_leftSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg, 0), endpoint = False)
                                self.line_matrix[self.line_rightSide_down] = 1
                                self.line_matrix[self.line_leftSide_down] = 1

                                self.cy_neg -= self.distance
                            except Exception as e:
                                break  
                        
                        self.compare_matrix = np.where(np.logical_and(self.line_matrix == 1, self.skeleton == 1), 1, 0)
                        
                        self.columns = []

                        for i  in range(0,self.compare_matrix.shape[0], self.distance):
                
                            if i==0:
                                self.columns.append(i)
                            else:
                                self.columns.append(i)
                                self.columns.append(-i)
                        

                        for j in range(0,len(self.compare_matrix)):
                            self.cond = False
                            for i in range(0, len(self.compare_matrix[j])):
                                if self.compare_matrix[j][i] == 1 and self.cond == False:
                                    self.cond = True
                                elif self.compare_matrix[j][i] == 1 and self.cond == True:
                                    self.compare_matrix[j][i] = 0

                                elif self.compare_matrix[j][i] == 0 and self.cond == True:
                                    self.cond = False
                                else:
                                    pass

                        
                        self.codes.append(self.filename)
                        self.all_data.append(self.create_data_for_lines(self.cy, self.cx, self.compare_matrix, self.distance, None, None, self.columns))
                        
                      
                    except Exception as e:
                        print(e)

                
                self.create_excel_lines(self.all_data, self.codes, self.columns)

                
            elif (self.ask_units.get() == 'microns'):
                self.all_data = []
                self.codes = []
                

                for file in os.listdir(self.directory_path):
                    try:
                        self.filepath = os.path.join(self.directory_path, file)
                        self.filename = os.fsdecode(file)

                        self.sholl = Sholl_analysis(self.filepath)
                        #self.skeleton = Skeleton(self.skeleton)
                        self.skeleton = self.sholl.skeleton()

                        self.cy, self.cx = self.sholl.get_soma_center()
                        self.cy_neg = round(self.cy)
                        self.cy_pos = round(self.cy)
                        self.center = np.array([self.cy, self.cx])

                        try:
                            self.distance = int(self.ask_distance.get("1.0", "end-1c"))
                        except:
                            self.distance = float(self.ask_distance.get("1.0", "end-1c"))

                        with tifffile.TiffFile(self.filepath) as tif:
                            self.tags = tif.pages[0].tags
                            self.resolution_x = self.tags.get('XResolution')

                            self.ratio_pixels_per_microns_x = self.resolution_x.value[0]/self.resolution_x.value[1]
                            self.ratio_microns_per_pixels_x = self.resolution_x.value[1]/self.resolution_x.value[0]

                        self.distance_in_pixels = int(round(self.ratio_pixels_per_microns_x * self.distance,0))

                        self.line_matrix = np.zeros((self.skeleton.shape[0],self.skeleton.shape[1]))

                        self.max_distance = self.skeleton.shape[0]

                        while self.cy_pos <= self.max_distance:
                            
                            try:
                                self.line_rightSide_up = line_nd((self.cy_pos, self.cx), (self.cy_pos,self.skeleton.shape[0]), endpoint = False)
                                self.line_leftSide_up = line_nd((self.cy_pos,self.cx), (self.cy_pos, 0), endpoint = False)
                                
                                self.line_matrix[self.line_rightSide_up] = 1
                                self.line_matrix[self.line_leftSide_up] = 1
                            
                                self.cy_pos += self.distance_in_pixels
                            except Exception as e:
                                break

                        
                        while self.cy_neg > 0:
                            
                            try:
                                
                                self.line_rightSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg,self.skeleton.shape[0]), endpoint = False)
                                self.line_leftSide_down = line_nd((self.cy_neg, self.cx), (self.cy_neg, 0), endpoint = False)
                                self.line_matrix[self.line_rightSide_down] = 1
                                self.line_matrix[self.line_leftSide_down] = 1

                                self.cy_neg -= self.distance_in_pixels
                            except Exception as e:
                                break  
                            
                        
                        self.compare_matrix = np.where(np.logical_and(self.line_matrix == 1, self.skeleton == 1), 1, 0)

                        self.columns = []

                        for i  in range(0,self.compare_matrix.shape[0], self.distance):
                
                            if i==0:
                                self.columns.append(i)
                            else:
                                self.columns.append(i)
                                self.columns.append(-i)
                        

                        for j in range(0,len(self.compare_matrix)):
                            self.cond = False
                            for i in range(0, len(self.compare_matrix[j])):
                                if self.compare_matrix[j][i] == 1 and self.cond == False:
                                    self.cond = True
                                elif self.compare_matrix[j][i] == 1 and self.cond == True:
                                    self.compare_matrix[j][i] = 0

                                elif self.compare_matrix[j][i] == 0 and self.cond == True:
                                    self.cond = False
                                else:
                                    pass


                        self.codes.append(self.filename) 
                        self.all_data.append(self.create_data_for_lines(self.cy, self.cx, self.compare_matrix, self.distance, self.ratio_pixels_per_microns_x, self.ratio_microns_per_pixels_x, self.columns))
                        self.columns = self.columns * self.ratio_microns_per_pixels_x

                    except Exception as e:
                        print(e)

                self.create_excel_lines(self.all_data, self.codes, self.columns)
        else:
            pass


    def run_analysis(self):

        #Condtions: if user did NOT fill required inputs, then output error, else run analysis   
        if len(self.ask_distance.get("1.0", "end-1c")) == 0:
            self.message_no_radius = tk.messagebox.showwarning(title = 'Error', message = "Type in distance.")


        elif self.var1.get() == 0 and self.var2.get() == 0:
            self.message_no_var = tk.messagebox.showwarning(title = 'Error', message = "Select one or many pictures.")

        elif (len(self.image_path) == 0) and (len(self.directory_path) == 0):
            self.message_no_path = tk.messagebox.showwarning(title = 'Error', message = "Select image or directory with images.")
            

        elif self.var2.get() == 1 and len(self.file_destination) == 0:
            self.message_no_destination = tk.messagebox.showwarning(title = 'Error', message = "Select destination folder.")

        elif self.var2.get() == 1 and len(self.ask_end_file_name.get("1.0", "end-1c")) == 0:
            self.message_no_excel_name = tk.messagebox.showwarning(title = 'Error', message = "Type excel result file name.")
            

        else:
            if (self.ask_type.get() == "Standard Sholl's"):
                self.run_analysis_circles()
            elif (self.ask_type.get() == 'Lines'):
                self.run_analysis_lines()

    def create_excel_lines(self, data, codes, columns):
        self.df = pd.DataFrame(data, columns = columns)
        self.df['Code'] = codes
        self.first_column = self.df.pop('Code')
        self.df.insert(0, 'Code', self.first_column)
        

        self.df.to_csv(self.file_destination + f"/{str(self.ask_end_file_name.get("1.0", "end-1c"))}_LinesResults.csv")
        self.done = tk.messagebox.showinfo('Complete', 'Analysis is done.')

    def create_excel(self, data, codes, columns):

        #Creating data frame and selecting separating "code" column
        self.df = pd.DataFrame(data, columns = columns)
        self.df['Code'] = codes
        self.first_column = self.df.pop('Code')
        
        #Initializing array, which will include values. max_columns contains radius at which max intersections was observed, last_column contains raidus where last intersections were observed
        self.max_column = []
        self.last_column = []

        

        for index, row in self.df.iterrows():
            self.max_of_row = 0
            self.max_of_column = ''
            self.last_of_col = ''

            self.prev_col = ''
            #Finding where the maximum intersections is located
            for column in self.df.columns:
                try:
                    if int(self.df._get_value(index, column)) > self.max_of_row:
                        self.max_of_row = self.df._get_value(index, column)
                        self.max_of_column = column
                    else:
                        continue
                except:
                    continue
            
            #Finding where the last intersection was
            for column in self.df.loc[:,self.max_of_column:].columns:
                try:
                    if int(self.df._get_value(index, column)) == 0:
                        self.last_of_col = self.prev_col
                    else:
                        self.prev_col = column
                except:
                    continue
            

            
            #appending max and last intersection of columns 
            self.max_column.append(self.max_of_column)
            self.last_column.append(self.last_of_col)

        #Inserting columns into dataframe
        self.df.insert(0, 'Code', self.first_column)
        self.df.insert(1, 'MaxIntersectionRadius',self.max_column)
        self.df.insert(2, 'LastRadius', self.last_column)
              
        self.df.to_csv(self.file_destination + f"/{str(self.ask_end_file_name.get("1.0", "end-1c"))}_ShollResults.csv")
        
        self.done = tk.messagebox.showinfo('Complete', 'Analysis is done.')



    def destination_file(self):
        self.file_destination = filedialog.askdirectory()
        

    def get_var1(self):
        if self.var1.get() == 1 and self.var2.get() == 1:
            self.var2.set(0)

    def get_var2(self):
        if self.var1.get() == 1 and self.var2.get() == 1:
            self.var1.set(0)

    def choose_file(self):
        if self.var1.get() == 1:
            self.image_path = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("Text files","*.txt*"),("all files","*.*")))
            
    
        elif self.var2.get() == 1:
            self.directory_path = filedialog.askdirectory()

    def create_data_for_lines(self, cy, cx, compare_matrix, distance, ratio_pixels_per_microns,ratio_microns_per_pixels, columns):
        if (self.ask_units.get() == 'pixels'):
            self.data = []
            self.cy_neg = round(cy)
            self.cy_pos = round(cy)
            self.center = np.array([cy,cx])



            for j in range(0, compare_matrix.shape[0]):
                
                if j ==0:
                    self.row_neg = compare_matrix[self.cy_neg,:]
                    self.data.append(self.row_neg.sum())
                    self.cy_neg += distance
                    self.cy_pos -= distance
                else:
                    try:
                        self.row_neg = compare_matrix[self.cy_neg,:]
                        self.row_pos = compare_matrix[self.cy_pos,:]
                        
                        self.data.append(self.row_pos.sum())
                        self.data.append(self.row_neg.sum())
                        

                        self.cy_neg += distance
                        self.cy_pos -= distance
                    except:
                        break

            
            for i in range(len(columns)-len(self.data)):
                self.data.append(np.int64(0))



            return self.data

        elif (self.ask_units.get() == 'microns'):
            
            self.data = []
            self.cy_neg = round(cy)
            self.cy_pos = round(cy)
            self.center = np.array([cy,cx])
                      

            for j in range(0, compare_matrix.shape[0]):
                
                if j ==0:
                    self.row_neg = compare_matrix[self.cy_neg,:]
                    self.data.append(self.row_neg.sum())
                    self.cy_neg -= int(round(distance * ratio_pixels_per_microns,0))
                    self.cy_pos += int(round(distance * ratio_pixels_per_microns,0))
                    
                else:
                    try:
                        self.row_neg = compare_matrix[self.cy_neg,:]
                        self.row_pos = compare_matrix[self.cy_pos,:]
                        
                        self.data.append(self.row_neg.sum())
                        self.data.append(self.row_pos.sum())
                        

                        self.cy_neg -= int(round(distance * ratio_pixels_per_microns,0))
                        self.cy_pos += int(round(distance * ratio_pixels_per_microns,0))
                        
                    except Exception as e:
                        break


            for i in range(len(self.columns)-len(self.data)):
                self.data.append(np.int64(0))

            return self.data
        
        else:
            pass


            

    
Masters()
