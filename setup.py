__author__ = 'chaganty'

from distutils.core import setup, Command

class UnitTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['pytest', '--doctest-modules',
                                 'obviousli', 'tests'])
        raise SystemExit(errno)

setup(
    name='obviousli',
    version='0.1',
    packages=['obviousli'],
    url='https://github.com/arunchaganty/obviousli',
    license='MIT',
    author='Stanford NLP',
    author_email='chaganty@cs.stanford.edu',
    description='A natural language inference system',
    cmdclass={'test': UnitTest},
    download_url='https://github.com/arunchaganty/obviousli/tarball/0.1',
    keywords=['nlp', 'neural networks', 'machine learning', 'logic', 'inference'],
)
