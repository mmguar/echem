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

__version__ = 0.1
def swv_load_timelapse(folder,frequency=10,potentiostat="Metrohm_simple",separ="\t",header=0,interval=300):

    separ_alt = '\s+'
    filename = '/'+frequency+'*'
    filenames = glob.glob(folder+filename)
    time_stamps=[]
    dict_times ={}
    filenames_sorted=natsorted(filenames)
    time_stamp=0
#for file in filenames:
    for file in filenames_sorted:  
        only_file = file[file.rfind('/'):][1:]
#    time_stamp = os.path.getctime(file)
#    if only_file == 'ignore' or (".nox" in only_file):
#        continue
        time_stamp+=interval
        time_stamps.append(time_stamp)

#    dict_times[time_stamp] = file
#
#time_stamps.sort(key=lambda time_stamp: float(time_stamp)) # sort based on time
#filenames_sorted = []
#for time_stamp in time_stamps:
#    filenames_sorted.append(dict_times[time_stamp])
#    print(dict_times[time_stamp][dict_times[time_stamp].rfind('/'):])
#
#
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

    for i, df in enumerate(dfs):
        time_stamps[i] = time_stamps[i] - start
        dfs[i]['Time Stamp'] =  time_stamps[i]   
        time_stamps_min[i] = time_stamps[i]/60
        dfs[i]['Time Stamp [min]'] =  time_stamps_min[i]   
        
    return dfs

def swv_timelapse_plot(dfs, interval=300):
    time_stamps = np.arange(len(dfs))*interval/60
    def interact_frame(time_stamp=0):
        plot_title = 'Time: ' + str(time_stamps[time_stamp])+' min'
        return hv.Points(
            data=dfs[time_stamp],
            kdims=[('Potential','Potential [V]'), ("Diff", "Current [A]")],
            vdims=['Time Stamp']
        ).opts(
        tools=['hover'],
        title = plot_title)
    
    plot = pn.interact(interact_frame, time_stamp =(0,len(time_stamps)-1))
    return plot

def swv_baseline_subtract(dfs,interval=300):
    for df in dfs:
        df["Peak Current"] = 0

    #eak_left = -0.34
    #eak_right = -0.28
    #eft_limit = 30
    #right_limit = 25
    time_stamps = np.arange(len(dfs))*interval/60

    steppp=dfs[0]["Potential"][0]-dfs[0]["Potential"][1]
    
    first_point = dfs[0]["Potential"].iloc[0]
    last_point = dfs[0]["Potential"].iloc[-1]
    
    peak_left_slider  = pnw.FloatSlider(name='Peak Left', value=-0.35, start=last_point, end=first_point, step=0.01)
    peak_right_slider  = pnw.FloatSlider(name='Peak Right', value=-0.25, start=last_point, end=first_point, step=0.01)
    limit_left_slider  = pnw.IntSlider(name='Limit Left', value=10, start=0, end=len(dfs[0]["Potential"])-int(0.1/steppp))
    limit_right_slider  = pnw.IntSlider(name='Limit Right', value=10, start=0, end=len(dfs[0]["Potential"])-int(0.1/steppp))
    time_step_slider  = pnw.IntSlider(name='Time Step', value=0, start=0, end=len(dfs)-1)
    @pn.depends(time_step_slider,peak_left_slider,peak_right_slider,limit_left_slider,limit_right_slider)
    def max_current(time_stamp=time_step_slider.param.value,a=peak_left_slider,b=peak_right_slider,c=limit_left_slider,d=limit_right_slider):
        return pn.pane.Markdown("peak current [A]: "+ str("{:.3e}".format(dfs[int(time_stamp)]["Peak Current"][0])))

    
    
    def update_sliders(time_stamp):
    
        peak_left_slider.start=float(decimal.Decimal(limit_left_slider.value*steppp+dfs[time_stamp]["Potential"].iloc[-1]+steppp).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_UP))
        #print(limit_left_slider.value*steppp+dfs[time_stamp]["Potential"].iloc[-1])
        #print(float(decimal.Decimal(limit_left_slider.value*steppp+dfs[time_stamp]["Potential"].iloc[-1]+steppp).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN)))
        #print(dfs[time_stamp]["Potential"].iloc[-(limit_left_slider.value+1)])
        peak_left_slider.end=peak_right_slider.value
        if (peak_left_slider.value > peak_left_slider.end):
            peak_left_slider.value = peak_left_slider.end
        if (peak_left_slider.value < peak_left_slider.start):
            peak_left_slider.value = peak_left_slider.start
        peak_right_slider.end=round(+dfs[time_stamp]["Potential"].iloc[0]-limit_right_slider.value*steppp,2)
        if (peak_right_slider.value > peak_right_slider.end):
            peak_right_slider.value = peak_right_slider.end
        if (peak_right_slider.value < peak_right_slider.start):
            peak_right_slider.value = peak_right_slider.start
        max_current.value=dfs[time_stamp]["Peak Current"][0]

    
    
    def interact_frame(time_stamp=0,peak_left=peak_left_slider.value,peak_right=peak_right_slider.value,left_limit=limit_left_slider.value,right_limit=limit_right_slider.value):
        
    
        peak_removed = [None] * len(dfs)
        removed = [None] * len(dfs)
        fit = [None]* len(dfs)
        baseline = [None]*len(dfs)

        
        
        for i, meas in enumerate(dfs):
    
            peak_removed[i] = dfs[i].loc[((dfs[i].Potential <= peak_left)) | (dfs[i].Potential >= peak_right ),\
                                        ['Potential', 'Diff']] 
            removed[i]= pd.concat([dfs[i].iloc[0:right_limit],dfs[i].iloc[-left_limit:]])#['Potential', 'Diff'])
            baseline[i]=pd.concat([peak_removed[i].iloc[right_limit:-left_limit], peak_removed[i].iloc[right_limit:-left_limit]])
            fit[i]=np.polynomial.polynomial.polyfit(peak_removed[i]['Potential'][right_limit:-left_limit], peak_removed[i]['Diff'][right_limit:-left_limit], 1)
            for j in dfs[i].index:
    
                dfs[i].at[j,'Fit'] = np.polynomial.polynomial.polyval(dfs[i].loc[j]['Potential'], fit[i])
                if (dfs[i].at[j,'Potential']>peak_left) and (dfs[i].at[j,'Potential']<peak_right):
                    dfs[i].at[j,'Peak'] = abs(dfs[i].loc[j]['Diff'] - dfs[i].loc[j]['Fit'])
            dfs[i]["Peak Current"] = dfs[i]['Peak'].max()

    
    
    
        
        plot_title = 'Time' + str(time_stamps[time_stamp])
        
        measurement_list   = [hv.Points(
            data=dfs[time_stamp],
             kdims=[('Potential','Potential [V]'), ('Diff','Current [A]')],
            vdims=[])]
    
        fit_list = [hv.Points(    
            data=dfs[time_stamp],
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
    
    
    
def swv_timelapse_timeplot(dfs, interval=300, sample_step=0, avg=0):
    
    time_stamps = np.arange(len(dfs))
    time_stamps_min = np.arange(len(dfs))*interval/60
    peak_currents = np.zeros(len(dfs))
    for i, df in enumerate(dfs):
        peak_currents[i] = df["Peak Current"][0]
    
    peaks = {"Time Stamp": time_stamps, "Peak": peak_currents, "Time Stamp [min]": time_stamps_min}

    df_peaks = pd.DataFrame (peaks, columns = ['Time Stamp','Peak',"Time Stamp [min]"])

    
    if avg:
        for df in dfs:
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
    else:
        return hv.Overlay(curve+points)