Capstone Project Commands

VM:
bhuvanvm: ssh msbhuvan@13.64.190.226
with_ssh_tunnel_for_Jupyter: ssh -N -f -L localhost:8889:localhost:8889 msbhuvan@13.64.190.226
stop connection: sudo netstat -lpn |grep :YYYY # and kill the shown PID.
starting jupyter: jupyter notebook --no-browser --port=8889 --allow-root


# Model training
The basic training code: https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13
