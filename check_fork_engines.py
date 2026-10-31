# coding: utf-8
"""本 fork 两路引擎的接线自检。

用途：合并上游之后跑一次，确认 `firedasr_onnx` / `crispasr_qwen3` 仍能被
`EngineFactory` 建出来。上游改了 `engines/base.py` 的能力枚举、`EngineFactory`
的注册表结构，或引擎 config 的 dataclass 字段名，这里都会立刻失败。

只读校验：不加载模型、不起子进程、不起 HTTP 服务。需要 numpy（CapsWriter 本来就有）。

    python check_fork_engines.py
"""
import dataclasses
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ENGINES = ('firedasr_onnx', 'crispasr_qwen3')
ARGS_NAMES = {'firedasr_onnx': 'FireRedOnnxArgs', 'crispasr_qwen3': 'CrispAsrQwen3Args'}
failures = []


def check(desc, ok, detail=''):
    print(f'[{"ok" if ok else "FAIL"}] {desc}' + (f' — {detail}' if detail else ''))
    if not ok:
        failures.append(desc)


import config_server as cs
from core.server.engines import factory
from core.server.engines.base import EngineCapabilities
from core.server.worker import check_model

check('config_server 与 engines.factory 可导入', True)

registry = set(factory.EngineFactory._ASR_LOADERS)
check('EngineFactory 注册表含两路引擎', set(ENGINES) <= registry, str(sorted(registry)))

for key in ENGINES:
    loader = getattr(factory.EngineFactory, f'_load_{key}', None)
    check(f'{key}: 工厂加载器存在', loader is not None)
    if loader is None:
        continue
    EngineClass, ConfigClass, ArgsObj = loader()
    cfg_args = getattr(cs, ARGS_NAMES[key], None)
    check(f'{key}: Args 与 config_server 里的是同一个类', ArgsObj is cfg_args,
          f'工厂拿到 {ArgsObj.__name__}，config_server 里是 '
          f'{cfg_args.__name__ if cfg_args else "不存在"}')

    fields = {f.name for f in dataclasses.fields(ConfigClass)}
    supplied = {k for k, v in ArgsObj.__dict__.items() if not k.startswith('_')}
    check(f'{key}: Args 没有 dataclass 拒收的多余字段', not (supplied - fields),
          f'多余 {sorted(supplied - fields)}')
    check(f'{key}: dataclass 必填字段都能由 Args 提供',
          not [f.name for f in dataclasses.fields(ConfigClass)
               if f.default is dataclasses.MISSING
               and f.default_factory is dataclasses.MISSING
               and f.name not in supplied])

    try:
        cfg = ConfigClass(**{k: v for k, v in ArgsObj.__dict__.items() if not k.startswith('_')})
        check(f'{key}: config 构造成功', True)
    except TypeError as e:
        check(f'{key}: config 构造成功', False, repr(e))
        continue

    paths = [f.name for f in dataclasses.fields(ConfigClass)
             if isinstance(getattr(cfg, f.name, None), str) and '/' in getattr(cfg, f.name)]
    for name in paths:
        v = getattr(cfg, name)
        looks_like_path = any(s in name for s in ('dir', 'path', 'exe'))
        if looks_like_path:
            check(f'{key}: {name} 是绝对 posix 路径', os.path.isabs(v), v)

    caps_src = inspect.getsource(EngineClass)
    for cap in ('ASR', 'PUNC'):
        check(f'{key}: 声明了 {cap}', f'EngineCapabilities.{cap}' in caps_src)

    check(f'{key}: check_model 有对应分支', key in inspect.getsource(check_model))

print()
if failures:
    print(f'{len(failures)} 项失败：')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('全部通过。合并上游后如果这里报 FAIL，先别 push，去对 '
      'docs/引擎扩展-FireRed与CrispASR.md 的接口约定。')
