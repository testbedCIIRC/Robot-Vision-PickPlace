# Environment set up
<br />
<b>1.</b> Clone this repository
<br/><br/>
<b>2.</b> Create a virtual environment 
<pre>
python -m venv robot_cell_env
</pre> 
<br/>
<b>3.</b> Activate virtual environment
<pre>
source robot_cell_env/bin/activate # Linux
.\robot_cell_env\Scripts\activate # Windows 
</pre>
<br/>
<b>4.</b> Add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=robot_cell_env
</pre>
<br/>
<b>5.</b> Run cells in rob_env_create.ipynb. Make sure environment kernel is selected.
