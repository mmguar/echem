import glob
import os
import time
import decimal 
from natsort import natsorted
import pandas as pd
import numpy as np

import panel as pn
import panel.widgets as pnw
import holoviews as hv
import bokeh.io


hv.extension('bokeh')
br = hv.renderer('bokeh')

__version__ = 0.2

class SWVTimelapse:
    
    def __str__(self):
        description =  "Dataframe of "+str(len(self.dfs))+" measurements, found at: "+str(self.folder)+"\nPotentiostat: "+self.potentiostat+".\nFrequency: "+str(self.frequency)+" Hz.\nInterval: "+str(self.interval)+" s."
        return (description)
    
    def delete(self, to_be_deleted):
        if type(to_be_deleted) is int: 
            to_be_deleted = [ to_be_deleted ]
        elif type(to_be_deleted) is not list:
            return
        
        for deletion in to_be_deleted:
            del self.dfs[deletion]
            self.time_stamps = np.delete(self.time_stamps, deletion)
            try:
                self.peak_currents= np.delete(self.peak_currents, deletion)
            except:
                continue
    
    def __init__(self, folder, potentiostat, separ='\t',frequency=10,interval=300):
        self.folder = folder
        self.interval = interval
        self.frequency = frequency
        self.potentiostat = potentiostat
        header=0
        filename = '/'+frequency+'*'
        filenames = glob.glob(folder+filename)
        time_stamps=[]
        dict_times ={}
        filenames_sorted=natsorted(filenames)
        time_stamp=0
        for file in filenames_sorted:  
            only_file = file[file.rfind('/'):][1:]
            time_stamp+=interval
            time_stamps.append(time_stamp)

        print("First file: "+str(filenames_sorted[0]))
        print("Last file: "+str(filenames_sorted[-1]))
        print("Number of files: "+str(len(filenames_sorted)))
    
        dfs = []
        for i,file in enumerate(filenames_sorted):
            dfs.append(pd.read_csv(file, sep=separ, na_values='*', header=header))
            try:
                len(dfs[i].columns)==2
            except Exception:
                dfs[i] = pd.read_csv(file, sep=separ_alt, na_values='*', header=header)
            dfs[i].head()
            dfs[i].columns = ['Potential','Diff']
            
            
            
        start = time_stamps[0]  
        time_stamps_min = np.zeros(len(time_stamps))
        self.time_stamps = np.arange(len(dfs))*interval/60
        for i, df in enumerate(dfs):
            time_stamps[i] = time_stamps[i] - start
            dfs[i]['Time Stamp'] =  time_stamps[i]   
            time_stamps_min[i] = time_stamps[i]/60
            dfs[i]['Time Stamp [min]'] =  time_stamps_min[i]   
        self.dfs = dfs 

    def plot(self):
        def interact_frame(time_stamp=0):
            plot_title = 'Time: ' + str(self.time_stamps[time_stamp])+' min'
            return hv.Points(
                data=self.dfs[time_stamp],
                kdims=[('Potential','Potential [V]'), ("Diff", "Current [A]")],
                vdims=['Time Stamp']
            ).opts(
            tools=['hover'],
            title = plot_title)
        
        plot = pn.interact(interact_frame, time_stamp =(0,len(self.time_stamps)-1))
        return plot

    
    
    def timeplot(self, sample_step=0, avg=0):
        try:
            self.peak_currents
        except:
            print("ERROR: No baseline has been set, please run swv_baseline_subtract before this.")
            return -1
        time_stamps = np.arange(len(self.dfs))
        time_stamps_min = self.time_stamps
        peaks = {"Time Stamp": time_stamps, "Peak": self.peak_currents, "Time Stamp [min]": time_stamps_min}
        df_peaks = pd.DataFrame (peaks, columns = ['Time Stamp','Peak',"Time Stamp [min]"])

        if avg:
            for df in self.dfs:
                baseline = df_peaks['Peak'][0:avg].mean()
                df_peaks_normalized = df_peaks.copy()
                df_peaks_normalized['Peak'] = df_peaks['Peak']/baseline
                curve = [hv.Curve(    
                        data=df_peaks_normalized,
                    kdims=['Time Stamp [min]'],vdims=[('Peak','Fold Increase')]).opts(width=600,height=300,alpha=0.3)]
                points = [hv.Points(    
                    data=df_peaks_normalized,
                    kdims=['Time Stamp [min]',('Peak','Fold Increase')]).opts(width=600,height=300,
                    marker='+',size=7,tools=['hover'])]

        else:
            curve = [hv.Curve(    
                data=df_peaks,
                kdims=['Time Stamp [min]'],vdims=[('Peak','Fold Increase')]).opts(width=600,height=300,alpha=0.3)]
            points = [hv.Points(    
                data=df_peaks,
                kdims=['Time Stamp [min]',('Peak','Fold Increase')]).opts(width=600,height=300,
                marker='+',size=7,tools=['hover'])]

        if sample_step:
            sample_line = [hv.VLine(sample_step*interval/60).opts(color='red')]
            return hv.Overlay(curve+points+sample_line)
        return hv.Overlay(curve+points)
    
    
    def baseline(self):
        self.peak_currents = np.zeros(len(self.dfs))



        #eak_left = -0.34
        #eak_right = -0.28
        #eft_limit = 30


        steppp=self.dfs[0]["Potential"][0]-self.dfs[0]["Potential"][1]

        first_point = self.dfs[0]["Potential"].iloc[0]
        last_point = self.dfs[0]["Potential"].iloc[-1]

        peak_left_slider  = pnw.FloatSlider(name='Peak Left', value=-0.35, start=last_point, end=first_point, step=0.01)
        peak_right_slider  = pnw.FloatSlider(name='Peak Right', value=-0.25, start=last_point, end=first_point, step=0.01)
        limit_left_slider  = pnw.IntSlider(name='Limit Left', value=10, start=0, end=len(self.dfs[0]["Potential"])-int(0.1/steppp))
        limit_right_slider  = pnw.IntSlider(name='Limit Right', value=10, start=0, end=len(self.dfs[0]["Potential"])-int(0.1/steppp))
        time_step_slider  = pnw.IntSlider(name='Time Step', value=0, start=0, end=len(self.dfs)-1)
        @pn.depends(time_step_slider,peak_left_slider,peak_right_slider,limit_left_slider,limit_right_slider)
        def max_current(time_stamp=time_step_slider.param.value,a=peak_left_slider,b=peak_right_slider,c=limit_left_slider,d=limit_right_slider):
            return pn.pane.Markdown("peak current [A]: "+ str("{:.3e}".format(self.peak_currents[0])))


    
        def update_sliders(time_stamp):
        
            peak_left_slider.start=float(decimal.Decimal(limit_left_slider.value*steppp+self.dfs[time_stamp]["Potential"].iloc[-1]+steppp).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_UP))
            #print(limit_left_slider.value*steppp+dfs[time_stamp]["Potential"].iloc[-1])
            #print(float(decimal.Decimal(limit_left_slider.value*steppp+dfs[time_stamp]["Potential"].iloc[-1]+steppp).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN)))
            #print(dfs[time_stamp]["Potential"].iloc[-(limit_left_slider.value+1)])
            peak_left_slider.end=peak_right_slider.value
            if (peak_left_slider.value > peak_left_slider.end):
                peak_left_slider.value = peak_left_slider.end
            if (peak_left_slider.value < peak_left_slider.start):
                peak_left_slider.value = peak_left_slider.start
            peak_right_slider.end=round(+self.dfs[time_stamp]["Potential"].iloc[0]-limit_right_slider.value*steppp,2)
            if (peak_right_slider.value > peak_right_slider.end):
                peak_right_slider.value = peak_right_slider.end
            if (peak_right_slider.value < peak_right_slider.start):
                peak_right_slider.value = peak_right_slider.start
            max_current.value=self.peak_currents[time_stamp]
    
        
        
        def interact_frame(time_stamp=0,peak_left=peak_left_slider.value,peak_right=peak_right_slider.value,left_limit=limit_left_slider.value,right_limit=limit_right_slider.value):
            
        
            peak_removed = [None] * len(self.dfs)
            removed = [None] * len(self.dfs)
            fit = [None]* len(self.dfs)
            baseline = [None]*len(self.dfs)
            
            
            for i, meas in enumerate(self.dfs):
        
                peak_removed[i] = self.dfs[i].loc[((self.dfs[i].Potential <= peak_left)) | (self.dfs[i].Potential >= peak_right ),\
                                            ['Potential', 'Diff']] 
                removed[i]= pd.concat([self.dfs[i].iloc[0:right_limit],self.dfs[i].iloc[-left_limit:]])#['Potential', 'Diff'])
                baseline[i]=pd.concat([peak_removed[i].iloc[right_limit:-left_limit], peak_removed[i].iloc[right_limit:-left_limit]])
                fit[i]=np.polynomial.polynomial.polyfit(peak_removed[i]['Potential'][right_limit:-left_limit], peak_removed[i]['Diff'][right_limit:-left_limit], 1)
                for j in self.dfs[i].index:
        
                    self.dfs[i].at[j,'Fit'] = np.polynomial.polynomial.polyval(self.dfs[i].loc[j]['Potential'], fit[i])
                    if (self.dfs[i].at[j,'Potential']>peak_left) and (self.dfs[i].at[j,'Potential']<peak_right):
                        self.dfs[i].at[j,'Peak'] = abs(self.dfs[i].loc[j]['Diff'] - self.dfs[i].loc[j]['Fit'])
                self.peak_currents[i] = self.dfs[i]['Peak'].max()
    
            
            plot_title = 'Time' + str(self.time_stamps[time_stamp])
            
            measurement_list   = [hv.Points(
                data=self.dfs[time_stamp],
                 kdims=[('Potential','Potential [V]'), ('Diff','Current [A]')],
                vdims=[])]
        
            fit_list = [hv.Points(    
                data=self.dfs[time_stamp],
                kdims=[('Potential','Potential [V]'), ('Fit','Current [A]')],
                vdims=[])]
            
            fit_list2 = [hv.Points(    
                data=peak_removed[time_stamp],
                 kdims=[('Potential','Potential [V]'), ('Diff','Current [A]')],
                vdims=[]).opts(color="r")]
            
            fit_list3 = [hv.Points(    
                data=removed[time_stamp],
                 kdims=[('Potential','Potential [V]'), ('Diff','Current [A]')],
                vdims=[]).opts(color="y")]
            
            fit_list4 = [hv.Points(    
            data=baseline[time_stamp],
            kdims=[('Potential','Potential [V]'), ('Diff','Current [A]')],
            vdims=[]).opts(color="purple")]
            
            overlay = hv.Overlay(measurement_list + fit_list + fit_list2 + fit_list3 + fit_list4)
            overlay = overlay.redim.label(x='Potential [V]', y='Current [A]')
            update_sliders(time_stamp)
            return overlay
        
        
    
        
        reactive_baseline = pn.bind(interact_frame, time_step_slider, peak_left_slider, peak_right_slider, limit_left_slider, limit_right_slider)
        
        widgets   = pn.Column("<br>\n## Baseline set:", peak_left_slider, peak_right_slider, limit_left_slider, limit_right_slider,max_current)
        occupancy = pn.Row(reactive_baseline, widgets)
        time_column = pn.Column("<br>\n## Time stamp:", time_step_slider)
        pn_overlay = pn.Column(time_column, occupancy)
        interact_frame()
        return pn_overlay
        
        