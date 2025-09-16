from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT ='-e'


def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements =file_obj.readline()
        [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name= 'California House Price',
    version= '0.0.1',
    author= 'Mithu',
    author_email= 'proshanta.mithu5@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages= find_packages(),
)