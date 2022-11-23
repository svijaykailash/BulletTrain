FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.9 python3-pip

# Install required Python packages
COPY reqt.txt .
RUN pip3 install -r reqt.txt

# Create a /work direcory within the container, copy everyhong from the 
# build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

# Make sure your scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh