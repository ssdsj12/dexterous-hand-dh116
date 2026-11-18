"""Device mapping for the independent preview, separate from Teacher GPU3 leases."""
import ctypes
import os
import re
import subprocess

def _pci(bus):
    if not isinstance(bus,str) or not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",bus):
        raise ValueError("invalid PCI device identity")
    return tuple(int(v,16) for v in re.split(r"[:.]",bus))

def validate_device(physical_gpu_index,*,mask,physical_bus,logical_bus,logical_count):
    if (type(physical_gpu_index) is not int or physical_gpu_index<0
            or mask!=str(physical_gpu_index) or type(logical_count) is not int or logical_count!=1
            or _pci(physical_bus)!=_pci(logical_bus)):
        raise ValueError("preview physical/logical GPU mapping differs")
    return physical_gpu_index

def probe_device(physical_gpu_index):
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,pci.bus_id",
        "--format=csv,noheader,nounits"],text=True,timeout=5)
    devices={int(parts[0].strip()):parts[1].strip()
        for line in raw.splitlines() if line.strip() for parts in [line.split(",")]}
    if physical_gpu_index not in devices:raise ValueError("selected preview GPU is absent")
    driver=ctypes.CDLL("libcuda.so.1")
    count=ctypes.c_int();device=ctypes.c_int();bus=ctypes.create_string_buffer(32)
    if (driver.cuInit(0)!=0 or driver.cuDeviceGetCount(ctypes.byref(count))!=0
            or driver.cuDeviceGet(ctypes.byref(device),0)!=0
            or driver.cuDeviceGetPCIBusId(bus,len(bus),device)!=0):
        raise ValueError("cannot read actual CUDA device identity")
    return validate_device(physical_gpu_index,mask=os.environ.get("CUDA_VISIBLE_DEVICES"),
        physical_bus=devices[physical_gpu_index],logical_bus=bus.value.decode("ascii"),logical_count=count.value)
