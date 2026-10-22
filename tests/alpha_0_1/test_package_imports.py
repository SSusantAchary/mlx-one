"""Public imports remain usable when native inference imports are forbidden."""

import subprocess
import sys


def test_public_imports_and_cli_help_do_not_import_native_backends():
    code = '''
import importlib.abc, sys
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'mlx', 'mlx_lm', 'mlx_vlm', 'mlx_audio', 'mlx_embeddings', 'transformers'
        }:
            raise AssertionError('eager inference import: ' + fullname)
sys.meta_path.insert(0, NoInference())
import mlx_one
from mlx_one.vision import load_vlm_model, stream_vlm
from mlx_one.audio import transcribe
from mlx_one.retrieval import embed, rerank
from click.testing import CliRunner
from mlx_one.cli import main
assert {'Engine', 'generate', 'stream_generate', 'load_text_model', 'transcribe',
        'embed', 'rerank', 'SFTTrainer', 'FastLanguageModel'} <= set(mlx_one.__all__)
assert CliRunner().invoke(main, ['--help']).exit_code == 0
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
