# python/eeg_processor.py
"""
EEG Data Processor
Reads EEG data from Arduino serial port (bio amplifier signal)
and converts to EDF format for epilepsy analysis
"""

import serial
import time
import numpy as np
from datetime import datetime
import pyedflib  
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys

SERIAL_PORT = 'COM7'  
BAUD_RATE = 115200

# EEG parameters
SAMPLING_RATE = 250  # Hz
CHANNEL_NAME = 'EEG Temporal'
PHYSICAL_MIN = -500  # Microvolts
PHYSICAL_MAX = 500   # Microvolts
DIGITAL_MIN = -32768
DIGITAL_MAX = 32767

OUTPUT_FOLDER = "EEG_Data_new"  

def connect_to_arduino():
    """Establish serial connection with Arduino"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        print(f"Connected to Arduino on {SERIAL_PORT}")
        time.sleep(2)  
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def collect_eeg_data(ser, duration_seconds=10):
    """Collect EEG data from Arduino for a specified duration"""
    print(f"Collecting EEG data for {duration_seconds} seconds...")
    
    start_time = time.time()
    data = []
    timestamps = []
    
   
    ser.flushInput()
    
    
    inside_buffer = False
    
    while time.time() - start_time < duration_seconds:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                
                
                if line == "BUFFER_START":
                    inside_buffer = True
                    continue
                elif line == "BUFFER_END":
                    inside_buffer = False
                    continue
                elif line == "EEG Monitoring System Initialized" or line == "Format: timestamp,eeg_value":
                    
                    continue
                
               
                if inside_buffer and ',' in line:
                    
                    ts, value = line.split(',')
                    data.append(float(value))
                    timestamps.append(float(ts))
                elif not inside_buffer and line.isdigit() or (line.replace('.', '', 1).isdigit() and line.count('.') < 2):
                    
                    data.append(float(line))
                    timestamps.append(time.time() - start_time)
            except (ValueError, UnicodeDecodeError) as e:
                
                print(f"Error parsing data: {e} for line: '{line}'")
                continue
    
    print(f"Collected {len(data)} samples")
    return np.array(data), np.array(timestamps)

def convert_to_microvolts(raw_data):
    """Convert ADC values to microvolts based on Arduino's 10-bit ADC"""
    conversion_factor = 500 / 1023
    return raw_data * conversion_factor

def preprocess_eeg_data(data):
    """Apply basic preprocessing to EEG data"""
    
    if len(data) == 0:
        print("Warning: No data to process")
        return np.array([0])
        
   
    data_uv = convert_to_microvolts(data)
    
    
    data_uv = data_uv - np.mean(data_uv)
    
    
    b, a = signal.butter(4, [0.5/(SAMPLING_RATE/2), 45/(SAMPLING_RATE/2)], btype='band')
    filtered_data = signal.filtfilt(b, a, data_uv)
    
    
    b_notch, a_notch = signal.iirnotch(60, 30, SAMPLING_RATE)
    filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)
    
    return filtered_data

def ensure_output_folder():
    """Create output folder if it doesn't exist"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")
    return os.path.abspath(OUTPUT_FOLDER)

def save_to_edf(data, patient_id="X", recording_info=None, file_name=None):
    """Save EEG data to EDF format"""
    output_dir = ensure_output_folder()
    
    if file_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"eeg_{patient_id}_{current_time}.edf"
    
   
    full_path = os.path.join(output_dir, file_name)
    
    n_channels = 1
    channel_info = {'label': CHANNEL_NAME, 
                   'dimension': 'uV', 
                   'sample_rate': SAMPLING_RATE,
                   'physical_min': PHYSICAL_MIN,
                   'physical_max': PHYSICAL_MAX,
                   'digital_min': DIGITAL_MIN,
                   'digital_max': DIGITAL_MAX,
                   'prefilter': 'HP:0.5Hz LP:45Hz Notch:60Hz',
                   'transducer': 'Ag/AgCl electrode'}
    
    
    with pyedflib.EdfWriter(full_path, n_channels=n_channels) as f:
        f.setPatientCode(patient_id)
        f.setPatientName(patient_id)
        
        
        if recording_info and 'start_datetime' in recording_info:
            f.setStartdatetime(recording_info['start_datetime'])
        else:
            f.setStartdatetime(datetime.now())
            
        f.setEquipment("Arduino + Bio Amplifier ExG Pill")
        f.setPhysicalMaximum(0, PHYSICAL_MAX)
        f.setPhysicalMinimum(0, PHYSICAL_MIN)
        f.setDigitalMaximum(0, DIGITAL_MAX)
        f.setDigitalMinimum(0, DIGITAL_MIN)
        f.setLabel(0, channel_info['label'])
        f.setPhysicalDimension(0, channel_info['dimension'])
        f.setPrefilter(0, channel_info['prefilter'])
        f.setTransducer(0, channel_info['transducer'])
        f.setSamplefrequency(0, SAMPLING_RATE)
        
       
        f.writeSamples([data])
    
    print(f"EEG data saved to {full_path}")
    return full_path

def analyze_eeg_for_epilepsy(data, sampling_rate=SAMPLING_RATE):
    """
    Basic feature extraction for epilepsy detection
    This is a simplified approach and not for clinical use
    """
   
    if len(data) < sampling_rate:
        print("Warning: Data too short for reliable analysis")
        return {
            'band_powers': {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0},
            'theta_alpha_ratio': 0,
            'spike_count': 0,
            'spike_rate': 0
        }
    
    
    freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(sampling_rate, len(data)))
    
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    
    band_powers = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band] = np.mean(psd[idx]) if np.any(idx) else 0
    
    
    theta_alpha_ratio = band_powers['theta'] / band_powers['alpha'] if band_powers['alpha'] != 0 else 0
    
    
    z_scores = (data - np.mean(data)) / np.std(data) if np.std(data) != 0 else np.zeros_like(data)
    spike_count = np.sum(np.abs(z_scores) > 3)  
    
    results = {
        'band_powers': band_powers,
        'theta_alpha_ratio': theta_alpha_ratio,
        'spike_count': spike_count,
        'spike_rate': spike_count / (len(data) / sampling_rate) if len(data) > 0 else 0  # Spikes per second
    }
    
    return results

def save_visualization(data, sampling_rate=SAMPLING_RATE, analysis_results=None, patient_id="Unknown", recording_info=None):
    """Save EEG visualization to file"""
    output_dir = ensure_output_folder()
    
    if len(data) == 0:
        print("No data to visualize")
        return None
        
    time_axis = np.arange(len(data)) / sampling_rate
    
    plt.figure(figsize=(12, 8))
    
   
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, data)
    
    title = f'EEG Signal (Temporal Lobe) - Patient {patient_id}'
    if recording_info and 'duration' in recording_info:
        title += f' - Duration: {recording_info["duration"]} seconds'
    plt.title(title)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True)
    
 
    plt.subplot(2, 1, 2)
    if len(data) >= sampling_rate:
        freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(sampling_rate, len(data)))
        plt.semilogy(freqs, psd)
        plt.title('Power Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (μV²/Hz)')
        plt.grid(True)
        
        
        if analysis_results:
            bands = {
                'delta': (0.5, 4, 'r'),
                'theta': (4, 8, 'g'),
                'alpha': (8, 13, 'b'),
                'beta': (13, 30, 'c'),
                'gamma': (30, 45, 'm')
            }
            
            for band, (low, high, color) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                if np.any(idx):
                    plt.fill_between(freqs[idx], psd[idx], color=color, alpha=0.3)
            
            plt.legend(['PSD'] + list(bands.keys()))
    else:
        plt.text(0.5, 0.5, 'Not enough data for spectrum analysis', 
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
   
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"eeg_plot_{patient_id}_{current_time}.png")
    plt.savefig(plot_filename)
    print(f"EEG visualization saved to {plot_filename}")
    
    return plot_filename
def save_analysis_report(analysis_results, patient_id="Unknown", recording_info=None):
    """Save analysis results to a text file"""
    output_dir = ensure_output_folder()
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(output_dir, f"eeg_analysis_{patient_id}_{current_time}.txt")
    
    with open(report_filename, 'w') as f:
        f.write(f"EEG Analysis Report for Patient {patient_id}\n")
        f.write(f"Analysis Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        
        if recording_info:
            if 'start_datetime' in recording_info:
                f.write(f"Recording Date and Time: {recording_info['start_datetime'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            if 'duration' in recording_info:
                f.write(f"Recording Duration: {recording_info['duration']} seconds\n")
            f.write("\n")
        
        f.write("Frequency Band Powers:\n")
        for band, power in analysis_results['band_powers'].items():
            f.write(f"{band.capitalize()}: {power:.6f}\n")
        
        f.write(f"\nTheta/Alpha ratio: {analysis_results['theta_alpha_ratio']:.2f}\n")
        f.write(f"Spike count: {analysis_results['spike_count']}\n")
        f.write(f"Spike rate: {analysis_results['spike_rate']:.2f} spikes/second\n")
        
       
        f.write("\nBasic Interpretation:\n")
        if analysis_results['theta_alpha_ratio'] > 1.5:
            f.write("- Elevated Theta/Alpha ratio may indicate abnormal brain activity\n")
        
        if analysis_results['spike_rate'] > 1.0:
            f.write("- Elevated spike rate detected\n")
            
        f.write("\nNote: This is a simplified analysis and should not be used for clinical diagnosis.\n")
        f.write("Please consult with a qualified medical professional for proper interpretation.\n")
    
    print(f"Analysis report saved to {report_filename}")
    return report_filename

def main(patient_id=None, duration_seconds=None):
    """Main function to run EEG analysis with optional command-line arguments"""

    ser = connect_to_arduino()
    if not ser:
        print("Failed to connect to Arduino. Please check connection and try again.")
        return
    
    try:
        
        output_dir = ensure_output_folder()
        print(f"EEG data will be saved to folder: {output_dir}")
        
        
        if patient_id is None:
            patient_id = input("Enter patient ID: ")
        
        
        if duration_seconds is None:
            duration_input = input("Enter recording duration in seconds (default: 60): ")
            try:
                duration_seconds = int(duration_input) if duration_input.strip() else 60
            except ValueError:
                print("Invalid duration. Using default value of 60 seconds.")
                duration_seconds = 60
        else:
            try:
                duration_seconds = int(duration_seconds)
            except ValueError:
                print("Invalid duration provided. Using default value of 60 seconds.")
                duration_seconds = 60
        
      
        start_datetime = datetime.now()
        print(f"\nStarting EEG recording for Patient {patient_id}")
        print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Recording Duration: {duration_seconds} seconds")
        
        
        recording_info = {
            'start_datetime': start_datetime,
            'duration': duration_seconds
        }
        
     
        data, timestamps = collect_eeg_data(ser, duration_seconds=duration_seconds)
        
        if len(data) == 0:
            print("No data collected. Please check your Arduino connection and code.")
            return
            
        
        preprocessed_data = preprocess_eeg_data(data)
        
       
        formatted_datetime = start_datetime.strftime("%Y%m%d_%H%M%S")
        edf_filename = f"eeg_{patient_id}_{formatted_datetime}.edf"
        edf_file = save_to_edf(preprocessed_data, patient_id, recording_info, edf_filename)
        
        
        analysis_results = analyze_eeg_for_epilepsy(preprocessed_data)
        
        
        print("\nEEG Analysis Results:")
        print(f"Frequency band powers: {analysis_results['band_powers']}")
        print(f"Theta/Alpha ratio: {analysis_results['theta_alpha_ratio']:.2f}")
        print(f"Spike count: {analysis_results['spike_count']}")
        print(f"Spike rate: {analysis_results['spike_rate']:.2f} spikes/second")
        
        
        report_file = save_analysis_report(analysis_results, patient_id, recording_info)
        
        
        plot_file = save_visualization(preprocessed_data, analysis_results=analysis_results, 
                                      patient_id=patient_id, recording_info=recording_info)
        
       
        print(f"\nAll data saved to folder: {output_dir}")
        print(f"EEG data file: {os.path.basename(edf_file)}")
        print(f"Analysis report: {os.path.basename(report_file)}")
        print(f"Plot file: {os.path.basename(plot_file)}")
        print("This data can now be used for more advanced epilepsy classification models")
        
        return {
            'edf_file': os.path.basename(edf_file),
            'report_file': os.path.basename(report_file),
            'plot_file': os.path.basename(plot_file)
        }
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return None
    finally:
        
        if ser and ser.is_open:
            ser.close()
            print("Serial connection closed")

if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
        duration_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        main(patient_id, duration_seconds)
    else:
        main()