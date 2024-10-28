from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
        This function will return the list of requirements from the file path
    '''
    requirements = []
    with open(file_path, 'r') as file:
        requirements = [req.replace('\n', '') for req in file.readlines()]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Stephane WAMBA',
    author_email='wambstephane@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)