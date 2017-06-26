from setuptools import setup

setup(
    name='sigrecog',
    version='1.0',
    entry_points={
        'console_scripts': [
            'sigrecog = __main__:main'
        ]
    },
    packages=['sigrecog'],
    license=open('LICENSE').read(),
    long_description=open('README.md').read(),
)
