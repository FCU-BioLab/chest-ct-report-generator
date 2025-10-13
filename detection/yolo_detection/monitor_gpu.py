#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time GPU Monitoring Script
"""

import subprocess
import time
import sys
from datetime import datetime

def get_gpu_info():
    """Get GPU utilization and memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        values = result.stdout.strip().split(',')
        gpu_util = int(values[0].strip())
        mem_used = int(values[1].strip())
        mem_total = int(values[2].strip())
        temp = int(values[3].strip())
        
        return gpu_util, mem_used, mem_total, temp
    except Exception as e:
        return None, None, None, None

def format_memory(mb):
    """Format memory in MB to GB"""
    return f"{mb / 1024:.2f} GB"

def get_color(value, thresholds):
    """Get color based on value and thresholds"""
    if value < thresholds[0]:
        return 'LOW'
    elif value < thresholds[1]:
        return 'NORMAL'
    else:
        return 'HIGH'

def main():
    print("=" * 80)
    print("GPU Real-time Monitor")
    print("=" * 80)
    print("Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            gpu_util, mem_used, mem_total, temp = get_gpu_info()
            
            if gpu_util is None:
                print("Error: Cannot get GPU info. Make sure nvidia-smi is available.")
                sys.exit(1)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            mem_percent = (mem_used / mem_total) * 100
            
            # Color indicators
            util_status = get_color(gpu_util, [30, 70])
            mem_status = get_color(mem_percent, [40, 80])
            temp_status = get_color(temp, [60, 75])
            
            # Print status
            if iteration % 20 == 0:
                print(f"\n{'Time':<10} {'GPU Usage':<15} {'Memory':<25} {'Temp':<12}")
                print("-" * 80)
            
            print(f"{timestamp:<10} "
                  f"{gpu_util:>3}% [{util_status:<6}] "
                  f"{format_memory(mem_used):>8} / {format_memory(mem_total):<8} ({mem_percent:>5.1f}%) [{mem_status:<6}] "
                  f"{temp}°C [{temp_status:<6}]")
            
            iteration += 1
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Monitoring stopped")
        print("=" * 80)
        sys.exit(0)

if __name__ == "__main__":
    main()
