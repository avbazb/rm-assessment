from setuptools import setup,find_packages
from glob import glob
setup(name='rmuc26_assessment',version='1.0.0',packages=find_packages(),
 data_files=[('share/ament_index/resource_index/packages',['resource/rmuc26_assessment']),('share/rmuc26_assessment',['package.xml']),('share/rmuc26_assessment/launch',glob('launch/*.py')),('share/rmuc26_assessment/worlds',glob('worlds/*.world')),('share/rmuc26_assessment/config',glob('config/*.rviz'))],
 entry_points={'console_scripts':['view=rmuc26_assessment.view:main']})
