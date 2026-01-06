#!/usr/bin/env python3
"""
REAL-TIME TRAINING MONITOR
Monitor GPU training progress and system resources
"""

import time
import os
import psutil
import GPUtil
from datetime import datetime
import curses
import threading
import subprocess

def get_training_progress():
    """Read current training progress from log file"""
    try:
        if os.path.exists('training.log'):
            with open('training.log', 'r') as f:
                lines = f.readlines()

            # Get last few lines for current status
            recent_lines = lines[-10:]

            current_epoch = 0
            current_loss = 0.0
            best_loss = float('inf')

            for line in recent_lines:
                if 'Epoch' in line and 'Train Loss' in line:
                    parts = line.split('|')
                    epoch_part = parts[0].strip()
                    loss_part = parts[1].strip()

                    # Extract epoch
                    epoch_text = epoch_part.split()[1]  # "Epoch X/Y"
                    current_epoch = int(epoch_text.split('/')[0])

                    # Extract loss
                    loss_text = loss_part.split(':')[1].strip()  # "Train Loss: X.XXXX"
                    current_loss = float(loss_text.split()[0])

                if 'Best model saved' in line:
                    # Extract best loss
                    loss_start = line.find('Loss:') + 6
                    loss_end = line.find(')', loss_start)
                    best_loss = float(line[loss_start:loss_end])

            return {
                'epoch': current_epoch,
                'current_loss': current_loss,
                'best_loss': best_loss,
                'lines': recent_lines[-3:]  # Last 3 log lines
            }
    except Exception as e:
        return {'error': str(e)}

    return {'epoch': 0, 'current_loss': 0.0, 'best_loss': float('inf'), 'lines': []}

def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory stats
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024**3)  # GB
        memory_total = memory.total / (1024**3)  # GB

        # GPU stats
        gpus = GPUtil.getGPUs()
        gpu_stats = []
        for i, gpu in enumerate(gpus):
            gpu_stats.append({
                'id': i,
                'name': gpu.name,
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                'temperature': gpu.temperature,
                'power_draw': gpu.powerDraw if hasattr(gpu, 'powerDraw') else 0,
                'power_limit': gpu.powerLimit if hasattr(gpu, 'powerLimit') else 0
            })

        # Disk stats
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024**3)  # GB
        disk_total = disk.total / (1024**3)  # GB

        # Network (basic)
        net = psutil.net_io_counters()
        net_sent = net.bytes_sent / (1024**2)  # MB
        net_recv = net.bytes_recv / (1024**2)  # MB

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used': memory_used,
            'memory_total': memory_total,
            'gpu_stats': gpu_stats,
            'disk_percent': disk_percent,
            'disk_used': disk_used,
            'disk_total': disk_total,
            'net_sent': net_sent,
            'net_recv': net_recv
        }

    except Exception as e:
        return {'error': str(e)}

def draw_monitor(stdscr):
    """Draw the monitoring interface"""
    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(1000)  # Refresh every second

    while True:
        stdscr.clear()

        # Get current stats
        training_stats = get_training_progress()
        system_stats = get_system_stats()

        # Title
        stdscr.addstr(0, 0, "🚀 GPU DOMINATION MODEL TRAINING MONITOR", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 60)

        # Training Progress
        stdscr.addstr(3, 0, "📊 TRAINING PROGRESS", curses.A_BOLD)
        stdscr.addstr(4, 0, f"Epoch: {training_stats.get('epoch', 0)}")
        stdscr.addstr(5, 0, f"Current Loss: {training_stats.get('current_loss', 0):.6f}")
        stdscr.addstr(6, 0, f"Best Loss: {training_stats.get('best_loss', float('inf')):.6f}")

        # Recent log lines
        stdscr.addstr(8, 0, "📝 RECENT LOGS:", curses.A_BOLD)
        log_lines = training_stats.get('lines', [])
        for i, line in enumerate(log_lines[-3:]):
            # Truncate long lines
            display_line = line.strip()[:58]
            stdscr.addstr(9 + i, 0, display_line)

        # System Resources
        stdscr.addstr(13, 0, "💻 SYSTEM RESOURCES", curses.A_BOLD)

        if 'error' not in system_stats:
            # CPU
            cpu_y = 14
            stdscr.addstr(cpu_y, 0, f"CPU: {system_stats.get('cpu_percent', 0):.1f}%")

            # Memory
            mem_y = 15
            stdscr.addstr(mem_y, 0, f"RAM: {system_stats.get('memory_used', 0):.1f}/{system_stats.get('memory_total', 0):.1f}GB ({system_stats.get('memory_percent', 0):.1f}%)")

            # GPU
            gpu_stats = system_stats.get('gpu_stats', [])
            if gpu_stats:
                gpu_y = 17
                stdscr.addstr(gpu_y, 0, "🔥 GPU STATS:", curses.A_BOLD)
                for i, gpu in enumerate(gpu_stats[:2]):  # Show max 2 GPUs
                    gpu_line = 18 + i
                    stdscr.addstr(gpu_line, 0, f"GPU{i}: {gpu['utilization']:.1f}% | "
                                                  f"{gpu['memory_used']}/{gpu['memory_total']}MB | "
                                                  f"{gpu['temperature']}°C")

            # Disk
            disk_y = 21
            stdscr.addstr(disk_y, 0, f"💾 Disk: {system_stats.get('disk_used', 0):.1f}/{system_stats.get('disk_total', 0):.1f}GB ({system_stats.get('disk_percent', 0):.1f}%)")

        else:
            stdscr.addstr(14, 0, f"System stats error: {system_stats['error']}")

        # Instructions
        stdscr.addstr(24, 0, "🎯 CONTROLS:", curses.A_BOLD)
        stdscr.addstr(25, 0, "Press 'q' to quit | 'r' to refresh | 'l' to show full logs")

        # Check for key press
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == ord('r'):
            continue  # Just refresh
        elif key == ord('l'):
            # Show full logs
            stdscr.clear()
            stdscr.addstr(0, 0, "📋 FULL TRAINING LOGS (last 20 lines)", curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * 50)

            try:
                with open('training.log', 'r') as f:
                    lines = f.readlines()[-20:]
                    for i, line in enumerate(lines):
                        display_line = line.strip()[:78]
                        stdscr.addstr(2 + i, 0, display_line)
            except:
                stdscr.addstr(2, 0, "Could not read training.log")

            stdscr.addstr(23, 0, "Press any key to return to monitor...")
            stdscr.getch()

        stdscr.refresh()
        time.sleep(1)

def start_training_monitor():
    """Start the training monitor in a separate thread"""
    def monitor_thread():
        try:
            curses.wrapper(draw_monitor)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Monitor error: {e}")

    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    return monitor

if __name__ == "__main__":
    print("🎥 Starting GPU Training Monitor...")
    print("Press Ctrl+C to stop monitoring")
    print()

    monitor = start_training_monitor()

    try:
        # Keep the main thread alive
        while monitor.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")

    print("Training monitor exited")
