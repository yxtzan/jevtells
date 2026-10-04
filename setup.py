"""Include editable repository config/i18n in wheels without duplicate sources."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root=Path(__file__).parent
        for name in ('config','i18n'):
            shutil.copytree(root/name,Path(self.build_lib)/'jevtells/data'/name,dirs_exist_ok=True)


setup(cmdclass={'build_py':BuildPy})
