import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.display import display, clear_output
from sklearn.linear_model import LinearRegression
import nidaqmx
from nidaqmx.constants import AcquisitionType
import datetime
import time
import serial
from scipy.signal import butter, filtfilt
import threading
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

stop_event = threading.Event()

def lowpass_filter(signal, fs, cutoff=1.0, order=5):
    nyquist = 0.5 * fs
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

class NIDAQ:
    def __init__(self, device_name="Dev2", channel="ai0"):
        self.device_name = device_name
        self.channel = channel
        self.measurement_number = 0
        #self.df_voltage = pd.DataFrame(columns=['Measurement', 'Voltage', 'Concentration [uM]', 'Time'])
        self.df_voltage = pd.DataFrame({
            'Measurement': pd.Series(dtype='int64'),
            'Voltage': pd.Series(dtype='float'),
            'Concentration [uM]': pd.Series(dtype='float'),
            'Time': pd.Series(dtype='datetime64[ns]')
        })
        self.slope = 0
        self.intercept = 0

    def turn_lamp_on(self):
        try:
            with nidaqmx.Task() as t:
                t.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao0", min_val=0.0, max_val=5.0)
                t.write(5.0)
                time.sleep(0.1)  # Allow time for the lamp to turn on
            print("Lamp turned ON.")
        except nidaqmx.DaqError as e:
            print(f"[DAQ Error] {e}")

    def turn_lamp_off(self):
        try:
            with nidaqmx.Task() as t:
                t.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao0", min_val=0.0, max_val=5.0)
                t.write(0.0)
                time.sleep(0.1)  # Allow time for the lamp to turn off
            print("Lamp turned OFF.")
        except nidaqmx.DaqError as e:
            print(f"[DAQ Error] {e}")

    def pulse_lamp(self):
        while not stop_event.is_set():
            try:
                with nidaqmx.Task() as t:
                    t.ao_channels.add_ao_voltage_chan("Dev2/ao0", min_val=0.0, max_val=5.0)
                    t.write(5.0, auto_start=True)
                    time.sleep(0.1)
                    t.write(0.0, auto_start=True)
            except nidaqmx.DaqError as e:
                print(f"[DAQ Error] {e}")
            time.sleep(0.9)
        print("Lamp pulsing stopped.")

    def save_calibration(self, filename="calibration_data.json"):
        if not hasattr(self, 'slope') or not hasattr(self, 'intercept'):
            print("⚠️ No calibration data found.")
            return
        data = {
            "slope": self.slope,
            "intercept": self.intercept,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"✅ Calibration data saved to {filename}")

    def load_calibration(self, filename="calibration_data.json"):
        if not os.path.exists(filename):
            print(f"⚠️ File not found: {filename}")
            return
        with open(filename, 'r') as f:
            data = json.load(f)
        self.slope = data["slope"]
        self.intercept = data["intercept"]
        print(f"✅ Calibration loaded: slope={self.slope}, intercept={self.intercept}")


    def perform_peak_voltage_detection(self, LSPone, sampling_rate=1000, idle_timeout=1.5, filter_cutoff=1.0):
        import plotly.graph_objects as go

        print(f"Waiting for pump to become active...")
        while True:
            LSPone.send_command('?6', with_trailing_R=False)
            response = LSPone.get_response()
            print(f"Pump response: {response}")
            if '@' in response:
                break
            time.sleep(0.1)

        print("Pump active. Starting continuous buffered acquisition...")

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.channel}",
                terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                min_val=0.0,
                max_val=10.0
            )
            task.timing.cfg_samp_clk_timing(
                rate=sampling_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1000
            )
            task.start()

            voltage_trace = []
            time_trace = []
            start_time = time.time()
            last_busy_time = start_time

            while True:
                LSPone.send_command('?6', with_trailing_R=False)
                response = LSPone.get_response()

                if '@' in response:
                    last_busy_time = time.time()
                elif time.time() - last_busy_time > idle_timeout:
                    print("Pump idle timeout reached. Stopping acquisition.")
                    break

                try:
                    voltages = task.read(number_of_samples_per_channel=1000, timeout=2.0)
                    now = time.time()
                    elapsed_start = now - start_time
                    timestamps = [elapsed_start + i / sampling_rate for i in range(len(voltages))]
                    voltage_trace.extend(voltages)
                    time_trace.extend(timestamps)
                except Exception as e:
                    print(f"DAQ read error: {e}")
                    break
        
        # lowpass filter
        filtered = lowpass_filter(voltage_trace, fs=sampling_rate, cutoff=filter_cutoff)

        baseline = np.mean(filtered[:int(3 * sampling_rate)])
        filtered -= baseline

        # 5-point moving average
        #smoothed = np.convolve(filtered, np.ones(5) / 5, mode='valid')
        #time_smoothed = time_trace[:len(smoothed)]
        smoothed = filtered
        time_smoothed = time_trace
        # Detect peak after 5 s
        time_np = np.array(time_smoothed)
        valid_indices = np.where(time_np > 1.0)[0]

        if len(valid_indices) > 0:
            peak_idx = np.argmax(smoothed[valid_indices])
            peak_voltage = smoothed[valid_indices][peak_idx]
            time_of_peak = time_np[valid_indices][peak_idx]
        else:
            peak_voltage = max(smoothed)
            time_of_peak = time_smoothed[smoothed.tolist().index(peak_voltage)]

        print(f"✅ Peak voltage: {round(peak_voltage, 4)} V at {round(time_of_peak, 2)} s")

        # Save
        current_time = datetime.datetime.now().time().replace(microsecond=0)
        df_new = pd.DataFrame({
            'Measurement': [self.measurement_number],
            'Voltage': [peak_voltage],
            'Concentration [uM]': [None],
            'Time': [current_time]
        })

        self.df_voltage = pd.concat([self.df_voltage, df_new], ignore_index=True)
        self.measurement_number += 1

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_smoothed, y=smoothed, mode='lines', name='Smoothed Voltage'))
        fig.add_trace(go.Scatter(
            x=[time_of_peak], y=[peak_voltage], mode='markers+text',
            marker=dict(color='red', size=10),
            text=[f'Peak: {peak_voltage:.3f} V'],
            textposition='top center',
            name='Peak'
        ))

        # Auto-tight y-axis using 1st–99th percentile
        v_min = np.percentile(smoothed, 1)
        v_max = np.percentile(smoothed, 99)

        fig.update_layout(
            title='Filtered & Smoothed Voltage Trace with Peak',
            xaxis_title='Time (s)',
            yaxis_title='Voltage (V)',
            xaxis=dict(range=[0, max(time_smoothed)]),
            yaxis=dict(range=[v_min - 0.01, v_max + 0.01])
        )

        plt.annotate(f"Peak: {peak_voltage:.4f} V", xy=(time_of_peak, peak_voltage),
                    xytext=(time_of_peak + 1, peak_voltage + 0.005),
                    arrowprops=dict(arrowstyle="->"), fontsize=10, color='red')

        plt.annotate(f"Baseline: {baseline:.4f} V", xy=(2, baseline),
                    xytext=(4, baseline + 0.005), fontsize=10, color='gray')

        plt.tight_layout()
        plt.close()

        return self.df_voltage, self.measurement_number, fig

    # Apply publication-style settings
    plt.style.use("default")
    mpl.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "legend.frameon": False
    })

    def calibration_curve_ignore_blank(self, show_plot=True):
        df_voltage = self.df_voltage.copy()

        # Drop NaNs and 'Blank' entries
        df_voltage = df_voltage.dropna(subset=["Voltage", "Concentration [uM]"])
        df_voltage = df_voltage[df_voltage["Concentration [uM]"] != "Blank"]

        if df_voltage.empty:
            print("No valid data points for calibration.")
            return None, None

        grouped = df_voltage.groupby('Concentration [uM]')['Voltage'].agg(['mean', 'std'])
        X = grouped.index.astype(float).values.reshape(-1, 1)
        y = grouped['mean'].values

        model = LinearRegression()
        model.fit(X, y)

        self.slope = model.coef_[0]
        self.intercept = model.intercept_
        r_squared = model.score(X, y)

        plt.figure(figsize=(7, 5))
        plt.errorbar(X.flatten(), y, yerr=grouped['std'], fmt='o', capsize=4,
                    markersize=6, linewidth=2, label='Mean ± SD')
        plt.plot(X, model.predict(X), linestyle='--', linewidth=2,
                label=f'Linear Fit (R² = {r_squared:.3f})')
        plt.xlabel('Concentration [µM]')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.tight_layout()
        
        if show_plot:
            plt.show()

        return model, df_voltage, plt

def estimate_concentration(self):
    df_voltage = self.df_voltage[self.df_voltage["Concentration [uM]"].isna()].copy()
    df_voltage["Estimated Concentration"] = (
        (df_voltage["Voltage"] - self.intercept) / self.slope
    ).apply(lambda x: round(x, 2))
    self.df_voltage.loc[df_voltage.index, "Estimated Concentration"] = df_voltage["Estimated Concentration"]
    return self.df_voltage

def remeasure_by_index(self, LSPone, meas_index, stock_concentration, max_A_stock=1000, fixed_A_water=1000):

    #Remeasure a specific entry in self.df_voltage by measurement number.

    try:
        old_row = self.df_voltage.loc[meas_index]
        concentration = float(old_row["Concentration [uM]"])
    except Exception as e:
        print(f"⚠️ Could not fetch measurement {meas_index}: {e}")
        return

    print(f"\n🔁 Remeasuring Measurement {meas_index} → {concentration} µM")

    # Generate dilution command
    ratio = concentration / stock_concentration
    A_stock = int(round(ratio * max_A_stock))
    if A_stock == max_A_stock:
        dilution_cmd = f'B2V200A{A_stock}B1V300A0R'
    else:
        dilution_cmd = f'B2V200A{A_stock}B12V200A{fixed_A_water}B1V300A0R'

    # Enzymes
    LSPone.send_command('B3V200A1500B1V300A0R')
    LSPone.wait_until_free()

    # Glucose + Water
    LSPone.send_command(dilution_cmd)
    LSPone.wait_until_free()

    # More Enzymes
    LSPone.send_command('B3V200A1500B1V300A0R')
    LSPone.wait_until_free()

    # Mix
    LSPone.send_command('B1gV500A3000V500A100M2000G3V500A100R')
    LSPone.wait_until_free()

    print("Waiting 3 min for reaction...")
    time.sleep(180)

    # Dispense to detection
    LSPone.send_command('B1V200A3000R')
    LSPone.wait_until_free()
    LSPone.send_command('B10V200A100R')

    # Measure
    df_voltage, meas_number, filenames = self.perform_peak_voltage_detection(LSPone)
    self.df_voltage.loc[df_voltage.index[-1], "Concentration [uM]"] = concentration
    self.df_voltage.loc[df_voltage.index[-1], "Note"] = f"Remeasure of {meas_index}"
    display(self.df_voltage.tail(1))

    # Wash
    LSPone.send_command('B1V200A3000R')
    LSPone.wait_until_free()
    LSPone.send_command('B11V200A0R')
    LSPone.wait_until_free()
    for _ in range(3):
        LSPone.send_command('B12V200A3000B10A100R')
        LSPone.wait_until_free()
    LSPone.send_command('B11V200A0R')
    LSPone.wait_until_free()


class AMF:
    def __init__(self, COM):
        self.COM = COM
        self.ser = serial.Serial(COM, baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)

    def send_command(self, command, with_trailing_R=True):
        time.sleep(0.25)
        self.ser.read_all()
        command_str = f'/1{command}'
        if with_trailing_R:
            command_str += 'R'
        command_str += '\r'
        print(f'Sending command: {command_str.strip()}')
        self.ser.write(command_str.encode())
        time.sleep(0.25)

    def get_response(self):
        response = self.ser.readline().decode().strip()
        return response

    def wait_until_free(self, idle_timeout=1.5):
        start_time = time.time()
        last_busy_time = time.time()
        while True:
            self.send_command('?6', with_trailing_R=False)
            response = self.get_response()
            if '@' in response:
                last_busy_time = time.time()
            else:
                if time.time() - last_busy_time > idle_timeout:
                    break
            time.sleep(1)

    def prime_tubes(self, tubes):
        for tube in tubes:
            command = f'B{tube}V200P2000R'
            print(f"Priming tube {tube} with command: {command}")
            self.send_command(command)
            self.wait_until_free()
            self.send_command('B1V200A0R')
            self.wait_until_free()

    def background_wash(self):
        for _ in range(3):
            self.send_command('B12V200A3000B2V200A100R')
            self.wait_until_free()
        self.send_command('B1V200A0R')
        self.wait_until_free()

    def wash(self):
        self.send_command('B12V200P3000R')
        self.wait_until_free()
        self.send_command('B2V200A100R')
        self.wait_until_free()

    def final_wash(self):
        for _ in range(6):
            self.send_command('B12V200P3000R')
            self.wait_until_free()
            self.send_command('B10V200A0R')
            self.wait_until_free()

    def generate_dilution_commands(self, concentrations, stock_conc, max_A_stock=1000, max_A_water=1000):
        command_dict = {}
        for conc in concentrations:
            ratio = conc / stock_conc
            A_stock = int(round(ratio * max_A_stock))
            A_water = max_A_water - A_stock
            if A_stock == max_A_stock:
                cmd = f'B4V200A{A_stock}B5A0R'
            else:
                cmd = f'B5V200P{A_water}B4P{A_stock}B5A0R'
            command_dict[conc] = cmd
        return command_dict




# For continuous glucose sampling
df_sample_results = pd.DataFrame(columns=['Measurement', 'Voltage', 'Estimated Concentration [uM]', 'Time'])
glucose_flow_rate = 1.5  # µL/min
stop_event = threading.Event()

def set_flow_rate(new_rate):
    global glucose_flow_rate
    glucose_flow_rate = new_rate
    print(f"✅ Flow rate updated to {glucose_flow_rate} µL/min")

def monitor_keyboard_stop():
    import sys
    print("🔁 Press 'q' then Enter at any time to stop.")
    while not stop_event.is_set():
        if input().strip().lower() == 'q':
            stop_event.set()
            print("🟥 Stop signal received. Finishing current cycle...")


def continuous_sample_estimation(daq, LSPone):
    global df_sample_results
    first_cycle = True
    stored_sample_ready = False 

    def sample_loop():
        nonlocal first_cycle
        nonlocal stored_sample_ready

        while not stop_event.is_set():
            print("\nNEW SAMPLE CYCLE STARTED")
            
            # Start glucose collection timer (threading 1)
            wait_time = int((12.5 / glucose_flow_rate) * 60)
            print(f"Starting glucose collection timer: {wait_time} sec")
            glucose_wait_done = threading.Event()
            threading.Timer(wait_time, glucose_wait_done.set).start()

            # Push enzymes into collection microtube
            print("Injecting 37.5 µL enzymes into collection tube...")
            LSPone.send_command('B3V200A2250B4V300A0R')
            LSPone.wait_until_free()

            #  First Cycle: Only collect and store sample

            if first_cycle:
                print("First cycle → waiting for sample to collect...")
                glucose_wait_done.wait()
                print("Sample collected. Aspirating to mixing tube...")
                LSPone.send_command('B4V200A3000B1V300A0R')
                LSPone.wait_until_free()
                stored_sample_ready = True
                first_cycle = False
                continue 

            #  Next Cycles: Process stored sample while new one collects

            if stored_sample_ready:
                # Mix stored sample
                print("🔁 Step 3: Mixing sample in mixing microtube...")
                LSPone.send_command('B1gV200A3000V200A100M2000G2V200A100R')
                LSPone.wait_until_free()

                # Start reaction + chip washing in background
                def reaction_wait_and_wash():
                    print("🧼 Step 4: Washing detection chip and waiting 3 min for reaction...")
                    for _ in range(3):
                        LSPone.send_command('B12V200A3000B10A100R')
                        LSPone.wait_until_free()
                    time.sleep(180)

                wash_thread = threading.Thread(target=reaction_wait_and_wash)
                wash_thread.start()
                wash_thread.join()

                # Inject to detection chip
                print("📤 Step 5: Injecting into detection chip...")
                LSPone.send_command('B1V200A3000R')
                LSPone.wait_until_free()
                LSPone.send_command('B10V200A100R')

                # Detect + estimate concentration
                print("📈 Step 6: Measuring and estimating...")
                df_voltage, meas_number, fig = daq.perform_peak_voltage_detection(LSPone)
                est_df = daq.estimate_concentration()
                latest = est_df.loc[est_df['Measurement'] == meas_number - 1].copy()
                latest = latest.rename(columns={"Estimated Concentration": "Estimated Concentration [uM]"})
                try:
                    df_sample_results.loc[len(df_sample_results)] = latest[["Measurement", "Voltage", "Estimated Concentration [uM]", "Time"]].values[0]
                    result = df_sample_results
                except Exception as e:
                    result = str(e)

                # Save figure
                last = df_voltage.loc[df_voltage["Measurement"] == meas_number - 1]
                conc = last["Concentration [uM]"].iloc[0] if "Concentration [uM]" in last else None
                est = last["Estimated Concentration"].iloc[0] if "Estimated Concentration" in last else None
                if isinstance(conc, (float, int)) and pd.notna(conc):
                    conc_label = f"{int(round(conc))}uM"
                elif isinstance(conc, str) and conc.lower() == "blank":
                    conc_label = "Blank"
                elif pd.notna(est):
                    conc_label = f"{int(round(est))}uM"
                else:
                    conc_label = "Unknown"
                filename = f"voltage_M{meas_number - 1}_{conc_label}.png"
                try:
                    fig.write_image(filename)
                    print(f"📁 Saved: {filename}")
                except Exception as e:
                    print(f"⚠️ Could not save figure: {e}")

                # Final rinse
                print("🚿 Final rinse...")
                LSPone.send_command('B1V200A1000R')
                LSPone.wait_until_free()
                LSPone.send_command('B11V200A0R')
                LSPone.wait_until_free()

            #Wait for glucose collection to finish
            print("⏳ Waiting for glucose timer to finish...")
            glucose_wait_done.wait()

            # Aspirate the newly collected sample to mixing tube
            print("🔄 Aspirating collected sample into mixing microtube...")
            LSPone.send_command('B4V200A3000B1V300A0R')
            LSPone.wait_until_free()
            stored_sample_ready = True  # ready for next cycle

    threading.Thread(target=monitor_keyboard_stop, daemon=True).start()
    sample_loop()


def get_sample_dataframe(show_plot=True):
    global df_sample_results

    if show_plot and not df_sample_results.empty:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        df_plot = df_sample_results.copy()
        df_plot['Time'] = pd.to_datetime(df_plot['Time'].astype(str))

        plt.figure(figsize=(8, 5))
        plt.plot(df_plot['Time'], df_plot['Estimated Concentration [uM]'], marker='o', linestyle='-')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.title("Estimated Glucose Concentration Over Time")
        plt.xlabel("Time")
        plt.ylabel("Estimated Concentration [µM]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_sample_results


__all__ = [
    'NIDAQ',
    'AMF',
    'stop_event',
    'set_flow_rate',
    'continuous_sample_estimation',
    'get_sample_dataframe'
]

