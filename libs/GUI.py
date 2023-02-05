import tkinter as tk
import serial
import serial.tools.list_ports
import threading
from pscad_translator import PSCAD_read
root = tk.Tk()
root.title("Serial Port Selector")

# Get list of available serial ports
ports = list(serial.tools.list_ports.comports())
port_names = [port.device for port in ports]

# Create a dropdown menu with the available ports
port_var = tk.StringVar()
port_var.set(port_names[0])
port_dropdown = tk.OptionMenu(root, port_var, *port_names)
port_dropdown.pack()

# Create a dropdown menu with the available baud rates
baud_rates = [115200,230400,460800,921600,2000000,3000000,4000000 ]
baud_var = tk.IntVar()
baud_var.set(115200)
baud_dropdown = tk.OptionMenu(root, baud_var, *baud_rates)
baud_dropdown.pack()

def receive_data(ser):
    while True:
        if ser.in_waiting:
            data = ser.read(15)
            handle_data(data)

def handle_data(data):
    terminal.insert(tk.END, str(data) + '\n')
    terminal.see(tk.END)
    print("Data received:", data)
    Rx1 = PSCAD_read(data)
    ser.write(Rx1)


def refresh_ports():
    global port_names
    global port_dropdown
    ports = list(serial.tools.list_ports.comports())
    port_names = [port.device for port in ports]
    port_dropdown.destroy()
    port_var.set(port_names[0])
    port_dropdown = tk.OptionMenu(root, port_var, *port_names)
    port_dropdown.pack()

def connect_serial():
    global ser
    port_name = port_var.get()
    baud_rate = baud_var.get()
    try:
        ser = serial.Serial(port = port_name,baudrate = baud_rate ,timeout=1,bytesize=8, stopbits=serial.STOPBITS_ONE)
        terminal.insert(tk.END, "Serial port connected." + '\n')
        receive_thread = threading.Thread(target=receive_data, args=(ser,))
        receive_thread.start()        
    except serial.SerialException:
        terminal.insert(tk.END, "Failed to open serial port\n")

def disconnect_serial():
    global ser
    if ser.is_open:
        ser.close()
        terminal.insert(tk.END, "Serial port closed." + '\n')
    else:
        root.destroy()
        print("Program closed.")

disconnect_button = tk.Button(root, text="Disconnect", command=disconnect_serial)
disconnect_button.pack()

refresh_button = tk.Button(root, text="Refresh", command=refresh_ports)
refresh_button.pack()

connect_button = tk.Button(root, text="Connect", command=connect_serial)
connect_button.pack()

def scroll_to_bottom(event):
    terminal.yview_moveto(1.0)


terminal = tk.Text(root)
terminal.pack(side=tk.LEFT,fill=tk.Y)
scrollbar = tk.Scrollbar(root,command=terminal.yview)
scrollbar.pack(side=tk.LEFT,fill=tk.Y)
terminal.config(yscrollcommand=scrollbar.set)
terminal.bind("<1>",scroll_to_bottom)

root.protocol("WM_DELETE_WINDOW", disconnect_serial)

root.mainloop()
