import ast
import copy
import importlib.util
import json
import multiprocessing
import multiprocessing.queues
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
spec = importlib.util.spec_from_file_location('queue_regression_schema', ROOT / 'core/server/schema.py')
schema = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schema
spec.loader.exec_module(schema)

class Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

def handler_method(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskHandler')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'handle_audio_task')
    namespace = {'deepcopy': copy.deepcopy, 'logger': Logger(), 'time': time}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<actual handler>', 'exec'), namespace)
    return namespace['handle_audio_task']

def queue_case(source):
    result = schema.Result(task_id='task', socket_id='socket', type='mic')
    result.tokens = ['first']
    result.timestamps = [1.0]
    state = types.SimpleNamespace(sessions={'task': object()}, get_session=lambda *args: types.SimpleNamespace(result=result))
    def fail(task):
        raise ValueError('failed segment')
    q = multiprocessing.Queue()
    obj = types.SimpleNamespace(pipeline=types.SimpleNamespace(process=fail), state=state, queue_out=q)
    task = types.SimpleNamespace(task_id='task', socket_id='socket', type='mic', time_start=1.0, time_submit=2.0, is_final=False)
    entered, release = threading.Event(), threading.Event()
    original = multiprocessing.queues._ForkingPickler.dumps
    def delayed(value, *args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError('feeder timeout')
        return original(value, *args, **kwargs)
    with patch.object(multiprocessing.queues._ForkingPickler, 'dumps', side_effect=delayed):
        try:
            handler_method(source)(obj, task)
            assert entered.wait(5)
            result.error = ''
            result.tokens.append('second')
            result.timestamps.append(2.0)
            result.is_final = True
        finally:
            release.set()
        received = q.get(timeout=5)
    q.close()
    q.join_thread()
    return received

old = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'eceeb93:core/server/worker/task_handler.py']).decode('utf-8')
new = (WORK / 'core/server/worker/task_handler.py').read_text(encoding='utf-8')
a, b = queue_case(old), queue_case(new)
assert a.error == '' and a.tokens == ['first', 'second'] and a.is_final
assert b.error == 'ValueError: failed segment' and b.tokens == ['first'] and b.timestamps == [1.0] and not b.is_final
print('QUEUE_REGRESSION_OK: original reproduced; snapshot preserved error, lists and final flag')

workflow = yaml.safe_load((ROOT / '.github/workflows/unbound-check.yml').read_text(encoding='utf-8'))
code = workflow['jobs']['pyright']['steps'][-1]['run']
compile(code, '<workflow>', 'exec')

def ci_case(name, paths=b'file with space.py\0', diagnostics=None, rc=0, git_failure=False, malformed=False, expected=0):
    calls = []
    def git(args):
        calls.append(args)
        assert '--diff-filter=ACMRT' in args and '-z' in args
        if git_failure:
            raise subprocess.CalledProcessError(128, args)
        return paths
    def pyright(args, **kwargs):
        calls.append(args)
        assert args[2:] == ['file with space.py']
        text = 'invalid' if malformed else json.dumps({'generalDiagnostics': diagnostics or []})
        return types.SimpleNamespace(returncode=rc, stdout=text, stderr='')
    status = 0
    with patch.dict(os.environ, {'BASE_SHA': 'a' * 40, 'EVENT_NAME': 'push'}, clear=False), patch('subprocess.check_output', side_effect=git), patch('subprocess.run', side_effect=pyright):
        try:
            exec(compile(code, '<workflow>', 'exec'), {})
        except SystemExit as exc:
            status = exc.code
        except (subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError):
            status = 1
    assert status == expected, (name, status, expected)
    if not paths or git_failure:
        assert len(calls) == 1
    print('CI_CASE_OK:', name)

def diag(rule):
    return {'rule': rule, 'severity': 'error', 'file': 'file with space.py', 'range': {'start': {'line': 0}}, 'message': rule}

ci_case('normal with spaced filename')
ci_case('git failure must fail', git_failure=True, expected=1)
ci_case('deleted-only excluded by diff filter', paths=b'')
ci_case('possibly unbound', diagnostics=[diag('reportPossiblyUnboundVariable')], rc=1, expected=1)
ci_case('definitely unbound', diagnostics=[diag('reportUnboundVariable')], rc=1, expected=1)
ci_case('unrelated import diagnostic allowed', diagnostics=[diag('reportMissingImports')], rc=1)
ci_case('tool infrastructure exit must fail', rc=2, expected=1)
ci_case('malformed output must fail', malformed=True, expected=1)
print('ALL_REVIEW_FIX_TESTS_OK')
