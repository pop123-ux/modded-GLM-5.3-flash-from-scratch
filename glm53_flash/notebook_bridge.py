"""Persistent Python objects in a separate interpreter, over stdin/stdout pipes.

Only run code you choose to execute. This is an execution helper, not a sandbox.
The host can load this module by file path without importing torch in its kernel.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

MARKER='__GLM_CELL_RESULT__'

class NotebookSession:
    def __init__(self,python,root):
        self.worker=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),'--worker'],
            cwd=str(root),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1,
            env=dict(os.environ,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','1')))
    def execute(self,source):
        if self.worker.poll() is not None:raise RuntimeError('Project session stopped; rerun its setup cell')
        request=uuid.uuid4().hex
        self.worker.stdin.write(json.dumps(dict(id=request,code=source))+'\n');self.worker.stdin.flush()
        try:
            for line in self.worker.stdout:
                if line.startswith(MARKER):
                    reply=json.loads(line[len(MARKER):])
                    if reply['id']!=request:raise RuntimeError('Unexpected project session reply')
                    if reply['status']=='error':raise RuntimeError('Project cell failed; see traceback above')
                    return reply
                print(line,end='',flush=True)
            raise RuntimeError('Project interpreter exited unexpectedly; rerun session setup')
        except KeyboardInterrupt:
            self.close()
            raise RuntimeError('Project session interrupted and stopped. Rerun setup and object-definition cells.') from None
    def close(self):
        if self.worker.poll() is None:
            self.worker.terminate()
            try:self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:self.worker.kill();self.worker.wait()
        for stream in (self.worker.stdin,self.worker.stdout):
            if stream:stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def worker():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    namespace={'__name__':'__main__'}
    def no_input(*args,**kwargs):raise RuntimeError('input() is unavailable in project cells; assign variables instead')
    namespace['input']=no_input
    for line in sys.stdin:
        request=json.loads(line);status='ok'
        try:
            tree=ast.parse(request['code'],filename='<glm-cell>')
            # Display the final expression like a notebook, while retaining objects.
            tail=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
            exec(compile(tree,'<glm-cell>','exec'),namespace)
            if tail is not None:
                value=eval(compile(ast.Expression(tail.value),'<glm-cell>','eval'),namespace)
                if value is not None:print(repr(value))
        except BaseException:
            status='error';traceback.print_exc()
        sys.stdout.write('\n'+MARKER+json.dumps(dict(id=request['id'],status=status))+'\n');sys.stdout.flush()

if __name__=='__main__':
    if sys.argv[1:]==['--worker']:worker()
    else:raise SystemExit('Use NotebookSession from the main experiment notebook')
