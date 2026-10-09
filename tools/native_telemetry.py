"""Bounded telemetry for one private benchmark process, with raw GPU samples."""
import csv,ctypes,io,json,os,subprocess,threading,time
from pathlib import Path
class Telemetry:
 def __init__(self,path):
  self.path=Path(path);self.pid=None;self.rows=[];self.stop=threading.Event()
  self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
 def _run(self):
  while not self.stop.is_set():
   row={'unix_s':time.time(),'pid':self.pid}
   try:
    p=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=0x08000000 if os.name=='nt' else 0)
    row['gpu_csv']=p.stdout.strip();row['gpu_returncode']=p.returncode
   except (OSError,subprocess.TimeoutExpired) as e:row['gpu_error']=str(e)
   if self.pid and os.name=='nt':
    class Mem(ctypes.Structure):
     _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong),('peak_ws',ctypes.c_size_t),('ws',ctypes.c_size_t),('peak_paged',ctypes.c_size_t),('paged',ctypes.c_size_t),('peak_nonpaged',ctypes.c_size_t),('nonpaged',ctypes.c_size_t),('pagefile',ctypes.c_size_t),('peak_pagefile',ctypes.c_size_t),('private',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Mem),ctypes.c_ulong]
    h=kernel.OpenProcess(0x410,False,self.pid)
    if h:
     m=Mem();m.cb=ctypes.sizeof(m)
     if psapi.GetProcessMemoryInfo(h,ctypes.byref(m),m.cb):row['process_memory']={k:getattr(m,k) for k in ('faults','peak_ws','ws','private','peak_pagefile')}
     kernel.CloseHandle(h)
   self.rows.append(row)
   with self.path.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
   self.stop.wait(1)
 def close(self):
  self.stop.set();self.thread.join(timeout=6)
  memories=[r['process_memory'] for r in self.rows if 'process_memory' in r]
  return {'samples':len(self.rows),'peak_working_set_bytes':max((m['peak_ws'] for m in memories),default=None),
          'peak_private_bytes_sampled':max((m['private'] for m in memories),default=None),
          'page_faults_note':'Windows total faults include soft faults; not a physical SSD counter',
          'gpu_note':'CSV is whole-device telemetry including desktop; fields: index,name,driver,memory used/total MiB,util%,SM/memory MHz,W,C'}
