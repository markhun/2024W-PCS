
# Use the official image as a parent image
FROM ubuntu:24.04

# Update the system, install OpenSSH Server, and set up users
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y openssh-server build-essential gcc libmpich-dev libopenmpi-dev dirmngr ca-certificates software-properties-common apt-transport-https dkms curl




RUN deluser --remove-home ubuntu
# Create user and set password for user and root user
RUN useradd -rm -d /home/dev -s /bin/bash -g root -G sudo -u 1000 dev && \
    echo 'dev:pass' | chpasswd && \
    echo 'root:pass' | chpasswd

RUN add-apt-repository -y ppa:graphics-drivers
RUN curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | sudo gpg --dearmor | sudo tee /usr/share/keyrings/nvidia-drivers.gpg > /dev/null 2>&1
RUN echo 'deb [signed-by=/usr/share/keyrings/nvidia-drivers.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /' | sudo tee /etc/apt/sources.list.d/nvidia-drivers.list
RUN apt-get update
RUN apt-get install --no-install-recommends -y nvidia-headless-570 cuda-cudart-12-6 cuda-compiler-12-6 cuda-command-line-tools-12-6 libnccl2=2.24.3-1+cuda12.6 libnccl-dev=2.24.3-1+cuda12.6 && \
    echo 'export CPATH=/usr/local/cuda/include/:$CPATH' >> /home/dev/.bashrc && \    
    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> /home/dev/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> /home/dev/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> /home/dev/.bashrc


# Set up configuration for SSH
RUN mkdir /var/run/sshd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    echo "export VISIBLE=now" >> /etc/profile

# Expose the SSH port
EXPOSE 22

# Run SSH
CMD ["/usr/sbin/sshd", "-D"]

