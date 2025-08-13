
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    ca-certificates \
    libgomp1 \
    libc6 \
    libgcc-s1 \
    libstdc++6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libxinerama1 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default python3 and create python symlink
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    ln -s /usr/bin/python3 /usr/bin/python

# Copy and install EnergyPlus 24.1.0 from local tools folder
COPY tools/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64.tar.gz /tmp/energyplus.tar.gz
RUN cd /tmp && \
    tar -xzf energyplus.tar.gz && \
    cp -r EnergyPlus-* /usr/local/EnergyPlus-24-1-0 && \
    rm -rf /tmp/*

# Add EnergyPlus to PATH
ENV PATH="/usr/local/EnergyPlus-24-1-0:$PATH"

WORKDIR /app

CMD ["/bin/bash"]