"""Race-tolerant storage measurement and PID-reuse-safe orphan supervision."""
import os,subprocess
from pathlib import Path

def process_sample(pid):
    try:raw=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:return None
    fields=raw.split(') ',1)[1].split()
    return {'start_ticks':int(fields[19]),'CPU_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),'state':fields[0],'threads':int(fields[17])}

def storage_bytes(paths,runner=subprocess.run):
    paths=list(map(str,paths));result=runner(['du','-sB1',*paths],text=True,capture_output=True);rows=[line.split(maxsplit=1) for line in result.stdout.splitlines()]
    if len(rows)!=len(paths) or any(len(row)!=2 or row[1]!=p for row,p in zip(rows,paths)):raise RuntimeError('storage measurement did not return all tracked roots')
    if result.returncode and (result.returncode!=1 or not result.stderr.strip() or any('No such file or directory' not in line for line in result.stderr.splitlines())):raise RuntimeError('storage measurement failed beyond transient removed-file race: '+result.stderr)
    return sum(int(row[0]) for row in rows)+105_000_000
