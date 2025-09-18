#!/usr/bin/env python3
"""
GPU monitoring script for document processing optimization.
Run this script to monitor GPU usage during API calls.
"""

import subprocess
import time
import csv
from datetime import datetime
import sys

def monitor_gpu(duration=300, interval=1):
    """
    Monitor GPU usage for specified duration.
    
    Args:
        duration: Monitoring duration in seconds (default: 5 minutes)
        interval: Monitoring interval in seconds (default: 1 second)
    """
    print(f"Starting GPU monitoring for {duration} seconds...")
    print("Press Ctrl+C to stop early")
    
    filename = f"gpu_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'GPU_Util%', 'Memory_Used_MB', 'Memory_Total_MB', 'Temp_C', 'Power_W'])
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                try:
                    result = subprocess.run([
                        'nvidia-smi', 
                        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        data = result.stdout.strip().split(', ')
                        timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
                        writer.writerow([timestamp] + data)
                        file.flush()
                        
                        # Print current status
                        gpu_util = data[0]
                        memory_used = data[1]
                        memory_total = data[2]
                        temp = data[3]
                        power = data[4]
                        
                        print(f"{timestamp}: GPU {gpu_util}%, Memory {memory_used}/{memory_total}MB, Temp {temp}°C, Power {power}W")
                    else:
                        print(f"Error getting GPU data: {result.stderr}")
                
                except subprocess.TimeoutExpired:
                    print("Timeout getting GPU data")
                except Exception as e:
                    print(f"Error: {e}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
    
    print(f"GPU monitoring data saved to: {filename}")
    return filename

def analyze_gpu_usage(filename):
    """
    Analyze GPU usage data and provide insights.
    
    Args:
        filename: Path to the CSV file with GPU data
    """
    print(f"\nAnalyzing GPU usage from {filename}...")
    
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)
        
        if not data:
            print("No data to analyze")
            return
        
        # Calculate statistics
        gpu_utils = [float(row['GPU_Util%'].replace('%', '')) for row in data]
        memory_used = [float(row['Memory_Used_MB']) for row in data]
        temps = [float(row['Temp_C']) for row in data]
        powers = [float(row['Power_W']) for row in data]
        
        # Basic statistics
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
        max_gpu_util = max(gpu_utils)
        avg_memory = sum(memory_used) / len(memory_used)
        max_memory = max(memory_used)
        avg_temp = sum(temps) / len(temps)
        max_temp = max(temps)
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        
        # Calculate active time
        active_time = sum(1 for util in gpu_utils if util > 5)  # >5% considered active
        total_time = len(gpu_utils)
        active_percentage = (active_time / total_time) * 100
        
        print(f"\n=== GPU Usage Analysis ===")
        print(f"Total monitoring time: {total_time} seconds")
        print(f"Active GPU time: {active_time} seconds ({active_percentage:.1f}%)")
        print(f"Idle time: {total_time - active_time} seconds ({100 - active_percentage:.1f}%)")
        print(f"\nGPU Utilization:")
        print(f"  Average: {avg_gpu_util:.1f}%")
        print(f"  Maximum: {max_gpu_util:.1f}%")
        print(f"\nMemory Usage:")
        print(f"  Average: {avg_memory:.0f} MB")
        print(f"  Maximum: {max_memory:.0f} MB")
        print(f"\nTemperature:")
        print(f"  Average: {avg_temp:.1f}°C")
        print(f"  Maximum: {max_temp:.1f}°C")
        print(f"\nPower Draw:")
        print(f"  Average: {avg_power:.1f}W")
        print(f"  Maximum: {max_power:.1f}W")
        
        # Performance assessment
        print(f"\n=== Performance Assessment ===")
        if active_percentage < 20:
            print("❌ POOR: GPU is mostly idle (<20% active time)")
            print("   Recommendation: Optimize batch processing and parallel operations")
        elif active_percentage < 50:
            print("⚠️  FAIR: GPU has moderate usage (20-50% active time)")
            print("   Recommendation: Increase batch sizes and parallel processing")
        elif active_percentage < 80:
            print("✅ GOOD: GPU has good usage (50-80% active time)")
            print("   Recommendation: Fine-tune batch sizes for optimal performance")
        else:
            print("🚀 EXCELLENT: GPU has excellent usage (>80% active time)")
            print("   Recommendation: Current configuration is well optimized")
        
        if max_gpu_util < 50:
            print("⚠️  Low peak GPU utilization - consider larger batch sizes")
        elif max_gpu_util > 90:
            print("⚠️  Very high peak GPU utilization - monitor for stability")
        
        if avg_memory < 2000:
            print("⚠️  Low memory usage - consider increasing batch sizes")
        elif avg_memory > 12000:
            print("⚠️  High memory usage - monitor for memory issues")
            
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze" and len(sys.argv) > 2:
            analyze_gpu_usage(sys.argv[2])
        else:
            duration = int(sys.argv[1]) if sys.argv[1].isdigit() else 300
            filename = monitor_gpu(duration)
            analyze_gpu_usage(filename)
    else:
        # Default: monitor for 5 minutes
        filename = monitor_gpu(300)
        analyze_gpu_usage(filename)
